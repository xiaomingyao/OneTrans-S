#!/bin/bash
# 在 122 容器内跑 OneTrans NPU：冒烟 + 精度 + 端到端训练脚本
set -u
SRC=/root/bytedance-onetrans/OneTrans_NPU
HOST=/mnt/paas/<工号>/bytedance-onetrans
CT=zeus-950-v2
PY=/opt/buildtools/torch_v2_pt2.7.1/bin/python

mkdir -p "$HOST"
tar czf - -C /root/bytedance-onetrans OneTrans_NPU | ssh 122 "mkdir -p $HOST && tar xzf - -C $HOST"
ssh 122 "docker exec $CT bash -lc 'rm -rf /tmp/onetrans_npu && mkdir -p /tmp/onetrans_npu'"
ssh 122 "docker cp $HOST/OneTrans_NPU/. $CT:/tmp/onetrans_npu/ >/dev/null"

echo "=========== 1) 冒烟（前向+反向+3 步）==========="
ssh 122 "docker exec $CT bash -lc 'cd /tmp/onetrans_npu && ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 $PY tests/smoke_npu.py --device npu'" 2>&1 | tail -12

echo "=========== 2) 精度（CPU fp32 golden vs NPU fp32 / bf16）==========="
ssh 122 "docker exec $CT bash -lc 'cd /tmp/onetrans_npu && ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 $PY tests/accuracy_npu.py --steps 10'" 2>&1 | tail -10

echo "=========== 3) 端到端训练（真数据集 TAAC2026/data_sample_1000，5 epoch）==========="
ssh 122 "docker exec $CT bash -lc 'cd /tmp/onetrans_npu && ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 HF_ENDPOINT=https://hf-mirror.com $PY scripts/run_taac2026_sample.py --device npu --batch-size 32 --epochs 5 --seq-len 32 --d-model 64 --multi-num 2 --num-pyramid-layers 3 --amp --amp-dtype bf16 --output-dir /tmp/onetrans_out'" 2>&1 | tail -18
