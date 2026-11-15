"""NPU 兼容层（新增文件，不改参考实现的模型逻辑）。

职责只有三件事：
  1. 设备：按 `--device` 决定是否 import torch_npu、设置当前 device、给出设备类型字符串；
  2. 混合精度：CUDA/NPU 各自的可用组合（NPU 只放行 bf16，fp16 需要 GradScaler 而 NPU 侧没有）；
  3. Scaler：CUDA fp16 用 torch.amp.GradScaler；NPU 用 no-op scaler（保持训练脚本调用形态不变）。

放在独立文件里，是为了让 `scripts/run_taac2026_sample.py` 的 diff 保持在"接线"级别，
模型代码（main_pytorch.py / models/）与参考实现逐字一致。
"""

from __future__ import annotations

import contextlib
import datetime
import os
from typing import Any

import torch


def init_device(device: str) -> str:
    """初始化设备并返回 device_type（'cpu' / 'cuda' / 'npu'）。"""
    device_type = torch.device(device).type
    if device_type == "npu":
        import torch_npu  # noqa: F401  # 必须在任何 npu 张量之前导入

        torch.npu.set_device(0)
    return device_type


def should_enable_amp(device_type: str, amp: bool, amp_dtype: str) -> bool:
    if not amp:
        return False
    if device_type not in ("cuda", "npu"):
        return False
    if amp_dtype == "fp16":
        # NPU 侧没有可用的 GradScaler 实现；fp16 需要 loss scaling 才能稳定，
        # 因此 NPU 只放行 bf16（bf16 动态范围与 fp32 相同，不需要 scaler）。
        return device_type == "cuda"
    if amp_dtype == "bf16" and device_type == "cuda":
        return torch.cuda.is_bf16_supported()
    return True


class _NoOpScaler:
    """NPU 路径的 scaler 占位：训练循环里 scale/step/update/unscale_ 全为空操作。"""

    def __init__(self, enabled: bool = False) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        return None

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None


def build_scaler(device_type: str, use_amp: bool, amp_dtype: torch.dtype) -> Any:
    if device_type == "npu":
        return _NoOpScaler(enabled=False)
    enabled = use_amp and device_type == "cuda" and amp_dtype == torch.float16
    return torch.amp.GradScaler(device=device_type, enabled=enabled)


def autocast_context(device_type: str, amp_dtype: torch.dtype, use_amp: bool):
    if not use_amp:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=True)


def describe(device_type: str) -> dict[str, str]:
    """给报告/日志用的版本指纹。"""
    info = {
        "device_type": device_type,
        "torch": torch.__version__,
    }
    if device_type == "npu":
        import torch_npu

        info["torch_npu"] = torch_npu.__version__
        info["soc"] = torch.npu.get_device_name(0)
    return info


# --------------------------------------------------------------------------
# 多卡（DDP over HCCL）—— 与生产 zeus-onetrans 对齐的三条约定
#   ① DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)
#   ② 训练循环里**不做**任何跨 rank 指标归约（生产侧只记录各 rank 本地 loss）
#   ③ 验证在 rank0 上跑全量，训练集按 rank 切分
# 反例（实测）：在训练 epoch 结束后紧接着发 all_gather/all_gather_object 汇总指标，
# 进程会停在 D2H 拷贝上一动不动（py-spy 栈在 `b.cpu()`），无报错、无输出。
# --------------------------------------------------------------------------


def init_distributed() -> tuple[int, int, int]:
    """按 torchrun 注入的环境变量初始化 HCCL 进程组，返回 (rank, world_size, local_rank)。

    `ASCEND_RT_VISIBLE_DEVICES` 已经把可见卡重编号为 0..N-1，因此 `npu:{LOCAL_RANK}` 即本进程该用的设备。
    单进程（WORLD_SIZE 缺省/为 1）返回 (0, 1, 0)，不做任何分布式初始化。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0

    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="hccl",
            timeout=datetime.timedelta(minutes=10),
        )
    return dist.get_rank(), world_size, local_rank


def dist_world_size() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def shutdown() -> None:
    """收尾：先同步设备再销毁进程组。

    不调用时实测：训练跑完后 rank1 在解释器退出阶段 SIGSEGV（exitcode -11），
    torchrun 因此报 ChildFailedError —— 训练本身是成功的，失败发生在 teardown。
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        try:
            if hasattr(torch, "npu"):
                torch.npu.synchronize()
            dist.destroy_process_group()
        except Exception:
            pass


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
