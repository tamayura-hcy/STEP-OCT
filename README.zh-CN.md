# STEP — 分阶段目标标签投影估计

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md)

把在一台扫描仪上训练好的 OCT 分类器，**无源、无梯度**地适配到另一台扫描仪的未标注数据上。

**论文：**《STEP: Staged Target-Label Estimation by Projection for Source-Free Cross-Device
OCT Classification》，投稿于 IEEE Transactions on Instrumentation and Measurement。
**代码：** https://github.com/tamayura-hcy/STEP-OCT

---

## 本仓库做了什么

一张 B-scan 不是组织的照片，而是一条测量链的输出；两台扫描仪就是同一个被测量的两条传递特性。
因此在一台设备上训练的分类器读到的是另一台仪器，而不是另一张视网膜。STEP 只在两处修补这一差异：

1. **在图像上。** 一个局部对比度均衡器，其强度由每张图像自身读出；随后在无标注目标队列上
   重估编码器的归一化缓冲区。
2. **在决策上。** 每张图像先读出一个临时标签（取自 stage-4 特征图的空间峰值位置），
   再在一个投影子空间内精化，子空间的秩按固定调度从 16 升到 80。

不需要源图像、不需要目标标签、没有梯度、没有优化器，也没有任何常数是对"即将被分类的队列"拟合的。
每张图像总共五次等效前向。

| | A→B | A→C | B→C |
|---|---|---|---|
| 冻结源 ResNet-50 | 57.06 | 70.68 | 61.76 |
| **STEP** | **92.92** | **90.65** | **90.39** |

（精度 %，五个种子的均值；详见 [`docs/RESULTS.md`](docs/RESULTS.md)）

## 安装

```bash
git clone https://github.com/tamayura-hcy/STEP-OCT.git
cd STEP-OCT
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

需要 Python ≥ 3.9、PyTorch ≥ 2.0。单张 12 GB 显卡足够，小规模队列用 CPU 也能跑。

## 数据

三份公开队列**不随仓库分发**。`docs/DATA.md` 给出了来源、许可与脚本预期的目录结构：

```
data/
  BOE/   {train,val,test}/{AMD,DME,NORMAL}/*.jpeg     # 2014_BOE_Srinivasan (A)
  TMI/   {train,val,test}/{AMD,DME,NORMAL}/*.jpeg     # Macular Dataset–Heidelberg (B)
  CELL/  {train,val,test}/{AMD,DME,NORMAL}/*.jpeg     # OCT2017 / ZhangLabData (C)
```

载入时只做一件事：短边缩放到 292、中心裁剪到 256×256；像素不再做其他处理。

## 快速开始

```bash
# 1. 为队列 A 训练冻结源模型
python scripts/train_source.py --data data/BOE/train --val data/BOE/val \
    --out runs/source_BOE.pth --epochs 30

# 2. 无标签、无梯度地适配到队列 B
python scripts/run_step.py --target data/TMI/test --ckpt runs/source_BOE.pth \
    --out runs/AB_step42.json --seed 42

# 3. 三个方向 × 五个种子，汇总成一张表
python scripts/evaluate.py --root data --ckpt-dir runs --out results/summary.json
```

第 2 步会打印精度、宏 one-vs-rest AUC 与逐类召回率。

## 目录结构

```
step/                 方法本体，六个短模块
  equalise.py         式 (1)：局部对比度均衡器，增益由队列测得
  views.py            四个视图：两个经过修正，两个是噪声实现
  bn.py               在无标注队列上重估归一化缓冲区
  extract.py          式 (2)：峰值描述子 H 与初始标签 T0
  readout.py          式 (3)：单一子空间内的渐进秩精化
  pipeline.py         run_step()：在一个目标队列上跑完整流程
  model.py, data.py   ResNet-50 + 分类头，ImageFolder 载入器
scripts/              train_source.py、run_step.py、evaluate.py
configs/default.yaml  共享配置的全部常数
docs/                 DATA.md、PROTOCOL.md、RESULTS.md
```

## 配置

`configs/default.yaml` 中的数字就是论文报告的数字，三个方向共用同一套：256×256 裁剪、批 64、
BN 动量 0.1、eps = 1e-6；3×3 视图的 tau = 0.30 ×（局部对比度中位数的通道均值），固定 5×5 视图
取 0.24；噪声 sigma = 0.007，来自两个各含 64 个张量的固定库；D = 80；秩 16 → 80，十个档位、每档三次更新。

峰值位置规则与秩调度是开发阶段在这些队列的带标签评估上选定的；`docs/PROTOCOL.md` 逐条记录了
"选了什么、在什么上选"，并且任何标签都不进入自适应更新。

## 引用

```bibtex
@article{han2026step,
  title={STEP: Staged Target-Label Estimation by Projection for Source-Free Cross-Device OCT Classification},
  author={Han, Chengyou and Luo, Yuemei and Liu, Weide and Cheng, Jun},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2026}
}
```

## 许可与数据

代码以 MIT 许可发布（见 `LICENSE`）。三份队列各自保留其原有许可，不随本仓库分发；
请向原始提供方获取。

## 联系

通讯作者：Yuemei Luo — luoyuemei@nuist.edu.cn
