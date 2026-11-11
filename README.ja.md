# STEP — 段階的ターゲットラベル投影推定

[English](README.md) | [中文](README.zh-CN.md) | [日本語](README.ja.md)

あるスキャナで学習済みの OCT 分類器を、**ソースデータなし・勾配なし**で別のスキャナの
ラベルなしデータに適応させます。

**論文：** "STEP: Staged Target-Label Estimation by Projection for Source-Free Cross-Device
OCT Classification"、IEEE Transactions on Instrumentation and Measurement 投稿。
**コード：** https://github.com/tamayura-hcy/STEP-OCT

---

## このリポジトリの内容

B-scan は組織の写真ではなく、計測チェーンの出力です。2 台のスキャナは同一の被測定量に対する
2 つの伝達特性であり、一方の装置で学習した分類器は「別の網膜」ではなく「別の計測器」を読んで
いることになります。STEP はその差を次の 2 か所だけで修復します。

1. **画像側。** 各画像から強度を読み取る局所コントラスト均等化と、ラベルなしターゲット
   コホート上でのエンコーダ正規化バッファの再推定。
2. **決定側。** 画像ごとに 1 つの暫定ラベルを stage-4 マップの空間ピーク位置で読み取り、
   ランクを 16 から 80 へ固定スケジュールで上げながら投影部分空間内で精緻化。

ソース画像なし、ターゲットラベルなし、勾配なし、オプティマイザなし、分類対象コホートに
対して調整した定数もなし。1 画像あたり画像等価で 5 回の順伝播のみです。

| | A→B | A→C | B→C |
|---|---|---|---|
| 凍結ソース ResNet-50 | 57.06 | 70.68 | 61.76 |
| **STEP** | **92.92** | **90.65** | **90.39** |

（精度 %、5 シードの平均。詳細は [`docs/RESULTS.md`](docs/RESULTS.md)）

## インストール

```bash
git clone https://github.com/tamayura-hcy/STEP-OCT.git
cd STEP-OCT
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9 以上、PyTorch 2.0 以上。12 GB の GPU 1 枚で足り、小規模コホートなら CPU でも動作します。

## データ

3 つの公開コホートは**再配布しません**。出典・ライセンス・スクリプトが想定する
ディレクトリ構成は `docs/DATA.md` を参照してください。

```
data/
  BOE/   {train,val,test}/{AMD,DME,NORMAL}/*.jpeg     # 2014_BOE_Srinivasan (A)
  TMI/   {train,val,test}/{AMD,DME,NORMAL}/*.jpeg     # Macular Dataset–Heidelberg (B)
  CELL/  {train,val,test}/{AMD,DME,NORMAL}/*.jpeg     # OCT2017 / ZhangLabData (C)
```

読み込み時に短辺 292 へリサイズし 256×256 に中心クロップするだけで、画素には他に何もしません。

## クイックスタート

```bash
# 1. コホート A の凍結ソースモデルを学習
python scripts/train_source.py --data data/BOE/train --val data/BOE/val \
    --out runs/source_BOE.pth --epochs 30

# 2. ラベルも勾配も使わずコホート B へ適応
python scripts/run_step.py --target data/TMI/test --ckpt runs/source_BOE.pth \
    --out runs/AB_step42.json --seed 42

# 3. 3 方向 × 5 シードを 1 つの表にまとめる
python scripts/evaluate.py --root data --ckpt-dir runs --out results/summary.json
```

手順 2 は精度、マクロ one-vs-rest AUC、クラス別再現率を出力します。

## リポジトリ構成

```
step/                 手法本体（6 つの短いモジュール）
  equalise.py         式 (1)：局所コントラスト均等化（ゲインはコホートから測定）
  views.py            4 ビュー：補正 2 つ + ノイズ実現 2 つ
  bn.py               ラベルなしコホートでの正規化バッファ再推定
  extract.py          式 (2)：ピーク記述子 H と初期ラベル T0
  readout.py          式 (3)：単一 subspaces 内での漸進ランク精緻化
  pipeline.py         run_step()：1 つのターゲットコホートで全体を実行
  model.py, data.py   ResNet-50 + ヘッド、ImageFolder ローダ
scripts/              train_source.py、run_step.py、evaluate.py
configs/default.yaml  共有設定のすべての定数
docs/                 DATA.md、PROTOCOL.md、RESULTS.md
```

## 設定

`configs/default.yaml` の値は論文で報告した値そのもので、3 方向で共通です：256×256 クロップ、
バッチ 64、BN モーメンタム 0.1、eps = 1e-6、3×3 ビューの tau = 0.30 ×（局所コントラスト
中央値のチャネル平均）、固定 5×5 ビューは 0.24、ノイズ sigma = 0.007（64 テンソルの固定バンク
2 つ）、D = 80、ランク 16 → 80 を 10 段階・各 3 回更新。

ピーク位置の規則とランクスケジュールは開発段階で同一コホートのラベル付き評価により選択した
ものです。`docs/PROTOCOL.md` に「何を何で選んだか」を記録しており、適応更新にラベルは入りません。

## 引用

```bibtex
@article{han2026step,
  title={STEP: Staged Target-Label Estimation by Projection for Source-Free Cross-Device OCT Classification},
  author={Han, Chengyou and Luo, Yuemei and Liu, Weide and Cheng, Jun},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2026}
}
```

## ライセンスとデータ

コードは MIT ライセンス（`LICENSE` 参照）。3 つのコホートはそれぞれのライセンスを保持し、
本リポジトリでは再配布しません。元の提供者から取得してください。

## 連絡先

責任著者：Yuemei Luo — luoyuemei@nuist.edu.cn
