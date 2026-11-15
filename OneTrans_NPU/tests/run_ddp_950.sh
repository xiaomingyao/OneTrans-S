#!/bin/bash
# 多卡（DDP/HCCL）验证与扩展性：等价性 → 1/2/4/8 卡固定全局 batch 步时
# 用法（122 上）：bash tests/run_ddp_950.sh [per_rank_batch]；数据集为 TAAC2026/data_sample_1000（与 GPU 同一数据集）
set -u
CT=zeus-950-v2
PY=/opt/buildtools/torch_v2_pt2.7.1/bin/python
HOST=/mnt/paas/<工号>/bytedance-onetrans
GB=${1:-64}

sync_in() {
  tar czf - -C "$(dirname "$0")/.." . | ssh 122 "rm -rf $HOST/OneTrans_NPU2 && mkdir -p $HOST/OneTrans_NPU2 && tar xzf - -C $HOST/OneTrans_NPU2 && rm -rf $HOST/OneTrans_NPU && mv $HOST/OneTrans_NPU2 $HOST/OneTrans_NPU"
  ssh 122 "docker exec $CT bash -lc 'rm -rf /tmp/onetrans_npu && mkdir -p /tmp/onetrans_npu' && docker cp $HOST/OneTrans_NPU/. $CT:/tmp/onetrans_npu/"
}

run() {  # $1=world_size $2=extra
  local w=$1; shift
  local lb=$((GB / w))
  local devs
  devs=$(seq -s, 0 $((w - 1)))
  echo "===== world_size=$w local_batch=$lb global_batch=$GB ====="
  ssh 122 "docker exec $CT bash -lc 'cd /tmp/onetrans_npu && ASCEND_RT_VISIBLE_DEVICES=$devs OMP_NUM_THREADS=1 \
    $PY -m torch.distributed.run --nproc_per_node=$w --master_port=2960$w \
    scripts/run_taac2026_sample.py --device npu --cache-dir /tmp/hf_cache \
    --epochs 8 --batch-size $lb --seq-len 32 --d-model 64 --multi-num 2 --num-pyramid-layers 3 \
    --amp --amp-dtype bf16 --output-dir /tmp/onetrans_ddp$w $*'" 2>&1 | grep -E '^\[(env|run|epoch|perf)\]|PASS|FAIL|Error|error' | head -20
}

sync_in

echo "########## 1) 梯度等价性（必须 PASS） ##########"
for w in 2 4; do
  ssh 122 "docker exec $CT bash -lc 'cd /tmp/onetrans_npu && ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((w-1))) $PY -m torch.distributed.run --nproc_per_node=$w --master_port=2955$w tests/ddp_equiv_npu.py --local-batch 8'" 2>&1 | grep -E '^\[A\]|^\[B\]|^ *单进程|verdict|DDP-EQUIV' | head -8
  ssh 122 "docker exec $CT bash -lc 'cd /tmp/onetrans_npu && ASCEND_RT_VISIBLE_DEVICES=$(seq -s, 0 $((w-1))) $PY -m torch.distributed.run --nproc_per_node=$w --master_port=2956$w tests/ddp_equiv_npu.py --local-batch 8 --dtype bf16'" 2>&1 | grep -E 'verdict|DDP-EQUIV' | head -3
done

echo "########## 2) 扩展性（固定全局 batch=$GB） ##########"
for w in 1 2 4 8; do
  if [ "$w" = "1" ]; then
    ssh 122 "docker exec $CT bash -lc 'cd /tmp/onetrans_npu && ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
      $PY scripts/run_taac2026_sample.py --device npu --cache-dir /tmp/hf_cache \
      --epochs 8 --batch-size $GB --seq-len 32 --d-model 64 --multi-num 2 --num-pyramid-layers 3 \
      --amp --amp-dtype bf16 --output-dir /tmp/onetrans_ddp1'" 2>&1 | grep -E '^\[epoch|^\[perf|^\[run\] device' | head -10
  else
    run "$w"
  fi
done
