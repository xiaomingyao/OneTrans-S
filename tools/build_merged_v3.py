#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 v13（三部分成稿）合并进 MIGRATION_BY_STAGE（五阶段 + GPU/NPU 并排）→ 一份 v3。

- 第一部分 = v13 §1 通用流程（表格形态）
- 第二部分 = v13 §2 DLRM（把「GPU/NPU 代码实现」转成并排面板围栏）
- 第三部分 = MIGRATION_BY_STAGE v2 的五阶段 onetrans（含并排面板），补 v13 的改动面/改动清单/原始日志/生产版对照
- 附录 A/B/C 取 v13（反查表、共同项、资产）
按评审要求：onetrans 不再有 7/8/9 节（五个坑 / 判据总表 / 边界）——v13 里同名的 3.5/3.8 不再搬回。
"""
import re

V13 = "/root/模型适配报告/迁移指导_流程与三部分_v13_20261114.md"
BY = "/root/bytedance-onetrans/docs/MIGRATION_BY_STAGE_onetrans_v2_20261114.md"
OUT = "/root/模型适配报告/pytorch迁移指导_v3_20261114.md"

v13 = open(V13, encoding="utf-8").read()
by = open(BY, encoding="utf-8").read()

# ------------------------------------------------------------------ 切分 v13
i1 = v13.index("## 第一部分")
i2 = v13.index("## 第二部分")
i3 = v13.index("## 第三部分")
iA = v13.index("### 附录 A")
part1 = v13[i1:i2].rstrip()
part2 = v13[i2:i3].rstrip()
p3 = v13[i3:iA].rstrip()
appendix = v13[iA:].rstrip()

# ------------------------------------------------------------------ DLRM：代码块 → 并排面板围栏
def compact(tag: str) -> str:
    """从标签里抽出 file:line 作为面板标题（面板标题会被 split 截断，故不留空格）。"""
    hits = re.findall(r"`([A-Za-z0-9_./]+\.py(?::[\d\-,]+)?)`", tag)
    if not hits:
        return ""
    return ",".join(h.split(":")[0] + (":" + h.split(":")[1] if ":" in h else "") for h in hits[:2])


PAIR = re.compile(r"\*\*(GPU|NPU) 代码实现\*\*（([^）]*)）\n\n```python\n(.*?)```\n\n"
                  r"\*\*(GPU|NPU) 代码实现\*\*（([^）]*)）\n\n```python\n(.*?)```", re.S)


def pair_repl(m):
    assert m.group(1) == "GPU" and m.group(4) == "NPU", m.group(0)[:60]
    gpath, npath = compact(m.group(2)), compact(m.group(5))
    return (f"```python gpu {gpath}\n{m.group(3)}```\n\n"
            f"```python npu {npath}\n{m.group(6)}```")


part2, npair = PAIR.subn(pair_repl, part2)
assert npair == 3, npair
assert "代码实现**（" not in part2, "还有没转的面板"

# ------------------------------------------------------------------ 从 v13 §3 抽要保留的块
def grab(start_marker, end_marker):
    a = p3.index(start_marker)
    b = p3.index(end_marker, a)
    return p3[a:b].rstrip()


rows_31 = [ln for ln in p3.split("\n")
           if ln.startswith("| **GPU 源码（源码地址）**") or ln.startswith("| **改动面**")
           or ln.startswith("| 改动占比")]
assert len(rows_31) == 3, rows_31

changes_tbl = grab("| 文件 | 改动行数 | 改点 | 执行逻辑 |", "\n\n### 3.3").rstrip()

smoke_log = grab("**① 冒烟**", "**② 精度**").rstrip()
e2e_log = grab("**③ 端到端**", "### 3.5").rstrip()
prod_cmp = grab("| 维度 | 本参考实现 | 生产 `zeus-onetrans/rerun-production` |", "\n\n### 3.8").rstrip()

# ------------------------------------------------------------------ 第三部分：BY_STAGE 正文重编号
body = by[by.index("## 0. 结论摘要"):].rstrip()

body = body.replace("## 0. 结论摘要", "### 3.1 结论摘要")
body = body.replace("## 1. 场景画像", "### 3.2 场景画像与改动面")
body = body.replace("## 2. 阶段一：环境与设备", "### 3.3 阶段一：环境与设备")
body = body.replace("## 3. 阶段二：稀疏特征与 Embedding —— 本模型 0 改动",
                    "### 3.4 阶段二：稀疏特征与 Embedding —— 本模型 0 改动")
body = body.replace("## 4. 阶段三：注意力与算子落点", "### 3.5 阶段三：注意力与算子落点")
body = body.replace("## 5. 阶段四：混合精度与显存", "### 3.6 阶段四：混合精度与显存")
body = body.replace("## 6. 阶段五：精度对齐与固化", "### 3.7 阶段五：精度对齐与固化")

SUB = {"2": "3.3", "4": "3.5", "5": "3.6", "6": "3.7"}
body = re.sub(r"^### (\d)\.(\d+) ",
              lambda m: f"#### {SUB[m.group(1)]}.{m.group(2)} " if m.group(1) in SUB else m.group(0),
              body, flags=re.M)
assert not re.search(r"^## \d\.", body, flags=re.M), "还有未重编号的一级节"

# §2 阶段二 → 新号
body = body.replace("所以 §2 阶段二的全部工作量为 0", "所以 §3.4 阶段二的全部工作量为 0")

# 3.2 场景画像：并入 v13 的改动面 / 改动占比 / GPU 源码行 + 改动清单表
body = body.replace("| 参考侧基线 | CPU fp32；GPU 侧未在本机测（无 GPU 卡），数值对照统一以 CPU fp32 作 golden |",
                    "| 参考侧基线 | CPU fp32；GPU 侧未在本机测（无 GPU 卡），数值对照统一以 CPU fp32 作 golden |\n"
                    + "\n".join(rows_31) + "\n\n#### 3.2.1 改动清单（逐文件）\n\n" + changes_tbl)

# 3.7 末尾追加原始日志 + 生产版对照
extra = ("\n\n### 3.8 验收原始日志（冒烟 / 端到端）\n\n"
         + smoke_log + "\n\n" + e2e_log + "\n\n---\n\n"
         + "### 3.9 与生产版 OneTrans 的差异（仅对照，生产版不在本篇范围）\n\n"
         + prod_cmp + "\n")
body = body + extra

# ------------------------------------------------------------------ 第一/二部分的交叉引用改到新号
part1 = part1.replace("| ① 环境与设备 | 兼容层 + `--device npu`（无平台档案）；种子补 NPU 段；多卡 `torchrun` + HCCL | 不需要档案；需要 hccl + **gloo 子组** | §3.3① / §2.3 |",
                      "| ① 环境与设备 | 兼容层 + `--device npu`（无平台档案）；种子补 NPU 段；多卡 `torchrun` + HCCL | 不需要档案；需要 hccl + **gloo 子组** | §3.3 / §2.3 |")
part1 = part1.replace("§3.3① / §2.3", "§3.3 / §2.3")
part1 = part1.replace("§3.2 / §2.4", "§3.4 / §2.4")
part1 = part1.replace("§3.3④ / §2.5", "§3.5.2 / §2.5")
part1 = part1.replace("§3.3② / §2.6", "§3.6 / §2.6")
part1 = part1.replace("§3.4 / §2.7", "§3.7 / §2.7")
part1 = part1.replace("同机实测（§3.4）", "同机实测（§3.5）")

part2 = part2.replace("（本场景由 sharder/planner 承担，onetrans 侧见 §3.3①）",
                      "（本场景由 sharder/planner 承担；onetrans 参考实现无稀疏表，不涉及）")

# ------------------------------------------------------------------ 附录：资产名更新
appendix = appendix.replace(
    "| 本篇 md / html（v13） | `/root/模型适配报告/迁移指导_流程与三部分_v13_20261114.{md,html}` |\n"
    "| v12（四部分：含生产版 zeus 章） | `/root/模型适配报告/迁移指导_流程与三场景_v12_20261114.{md,html}` |",
    "| **本篇 md / html（pytorch迁移指导 v3）** | `/root/模型适配报告/pytorch迁移指导_v3_20261114.{md,html}` |\n"
    "| 三部分成稿（v13，未并排） | `/root/模型适配报告/迁移指导_流程与三部分_v13_20261114.{md,html}` |\n"
    "| onetrans 单篇（v2，五阶段并排） | `/root/bytedance-onetrans/docs/MIGRATION_BY_STAGE_onetrans_v2_20261114.{md,html}` |")
appendix = appendix.replace(
    "| onetrans 迁移专项指导（按五阶段 + GPU/NPU 并排） | `/root/bytedance-onetrans/docs/MIGRATION_BY_STAGE_onetrans_v2_20261114.{md,html}` |\n", "")
appendix = appendix.replace("| onetrans 详细版（含跑测命令） |", "| onetrans 详细版（含跑测命令，未并排） |")

# ------------------------------------------------------------------ 头
HEAD = f"""# PyTorch 模型迁移昇腾：通用流程 + DLRM + onetrans（五阶段 × GPU/NPU 实现对照）

版本：v3 · 2026-11-14（**合并稿**：v13 三部分成稿 + onetrans 五阶段并排实现。§2/§3 每个改点给 **GPU 实现与 NPU 实现并排**；onetrans 部分按评审意见不含原 7/8/9 节（五个坑 / 判据总表 / 边界），v13 里同名的"五个坑/边界"也不搬回；单篇 v13、v2 保留）

**场景**：DLRM（Meta 原版 `dlrm_s_pytorch.py` + torchrec `dlrm_main.py`）· onetrans（`https://github.com/WestbrookLong/OneTrans_Pytorch`，commit `8febd7e`）
**环境**：Ascend950DT_9582 / CANN 9.1.0 / torch 2.7.1+cpu / torch_npu 2.7.1.post4 / Python 3.11.0
**组织方式**：三部分 —— §1 通用五阶段流程（做什么/判据/不要照抄的 5 处）→ §2 DLRM → §3 onetrans（五阶段逐点对照，GPU 与 NPU 并排）。
**判据口径**：行号均为本机可 `sed -n` 复核的真实位置；GPU 侧只作代码对照，不含 GPU 性能数据。

三部分怎么用：

| 部分 | 回答什么 | 读者 |
|---|---|---|
| **§1 通用适配流程** | 迁移分几个阶段、每阶段做什么、判据是什么、哪 5 处不能照抄通用教程 | 先读一遍，建立骨架 |
| **§2 DLRM 迁移指导** | 稀疏表 + MLP 的推荐模型具体改哪几行、按什么顺序、怎么判过（GPU 源码：[github.com/facebookresearch/dlrm](https://github.com/facebookresearch/dlrm)） | 做 DLRM 的人 |
| **§3 onetrans 迁移指导** | 序列建模的参考实现移植到 950 的裁剪流程、5 处改动、五阶段实测判据（GPU 源码：[github.com/WestbrookLong/OneTrans_Pytorch](https://github.com/WestbrookLong/OneTrans_Pytorch)） | 拿到开源实现要快速上板的人 |

![图7 五个阶段的适配改动：OneTrans 与 DLRM 分别改哪几处](/root/figs_pt_migration/fig7_adapt_flow_v9.png)

---
"""

PART3_HEAD = ("## 第三部分：onetrans 迁移指导（五阶段 × GPU/NPU 实现对照）\n\n"
              "> 本节按五阶段展开，每个改点给 **GPU 实现与 NPU 实现并排面板**；"
              "GPU 侧参考仓是单进程单卡（无 DDP/分布式代码），凡参考仓没有的写法均显式标注为「等价写法」。\n")
md = HEAD + part1 + "\n\n---\n\n" + part2 + "\n\n---\n\n" + PART3_HEAD + "\n" + body + "\n\n---\n\n" + appendix + "\n"
# 附录升为一级（不再挂在第三部分下面）
for L in ("A", "B", "C"):
    md = md.replace(f"### 附录 {L}：", f"## 附录 {L}：")
open(OUT, "w", encoding="utf-8").write(md)
print("OK ->", OUT, md.count("\n") + 1, "行，面板围栏", len(re.findall(r"```python (?:gpu|npu) ", md)))
