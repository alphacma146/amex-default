# American Express - Default Prediction

- [American Express - Default Prediction](#american-express---default-prediction)
  - [1. Abstract](#1-abstract)
  - [2. Data](#2-data)
  - [3. feather 形式に変換](#3-feather-形式に変換)
  - [4. データの概観](#4-データの概観)
    - [4.1. 明細数の確認](#41-明細数の確認)
    - [4.2. カテゴリ別に target を比較](#42-カテゴリ別に-target-を比較)
    - [4.3. パラメーター毎に target で比較](#43-パラメーター毎に-target-で比較)
      - [Delinquency variables](#delinquency-variables)
      - [Spend variables](#spend-variables)
      - [Payment variables](#payment-variables)
      - [Balance variables](#balance-variables)
      - [Risk variables](#risk-variables)
      - [パラメーターの分類](#パラメーターの分類)
    - [4.4. 相関行列と寄与率](#44-相関行列と寄与率)
      - [Delinquency variables](#delinquency-variables-1)
      - [Spend variables](#spend-variables-1)
      - [Payment variables](#payment-variables-1)
      - [Balance variables](#balance-variables-1)
      - [Risk variables](#risk-variables-1)
  - [5.1. LightGBM](#51-lightgbm)
    - [GPU 版インストール手順](#gpu-版インストール手順)
    - [dart](#dart)
    - [評価関数](#評価関数)
  - [6.1. 結果](#61-結果)
    - [顧客毎に最新の 2 明細で学習](#顧客毎に最新の-2-明細で学習)
    - [ハイパーパラメータチューニング](#ハイパーパラメータチューニング)
    - [主成分分析でパラメータを圧縮](#主成分分析でパラメータを圧縮)
    - [パラメータ毎に統計量を算出、カテゴリーパラメータは one-hot-encoding](#パラメータ毎に統計量を算出カテゴリーパラメータは-one-hot-encoding)
    - [主成分分析で時間軸方向に圧縮](#主成分分析で時間軸方向に圧縮)

---

## 1. Abstract

> Whether out at a restaurant or buying tickets to a concert, modern life counts on the convenience of a credit card to make daily purchases. It saves us from carrying large amounts of cash and also can advance a full purchase that can be paid over time. How do card issuers know we’ll pay back what we charge? That’s a complex problem with many existing solutions—and even more potential improvements, to be explored in this competition.
> Credit default prediction is central to managing risk in a consumer lending business. Credit default prediction allows lenders to optimize lending decisions, which leads to a better customer experience and sound business economics. Current models exist to help manage risk. But it's possible to create better models that can outperform those currently in use.
> American Express is a globally integrated payments company. The largest payment card issuer in the world, they provide customers with access to products, insights, and experiences that enrich lives and build business success.
> In this competition, you’ll apply your machine learning skills to predict credit default. Specifically, you will leverage an industrial scale data set to build a machine learning model that challenges the current model in production. Training, validation, and testing datasets include time-series behavioral data and anonymized customer profile information. You're free to explore any technique to create the most powerful model, from creating features to using the data in a more organic way within a model.
> If successful, you'll help create a better customer experience for cardholders by making it easier to be approved for a credit card. Top solutions could challenge the credit default prediction model used by the world's largest payment card issuer—earning you cash prizes, the opportunity to interview with American Express, and potentially a rewarding new career.

---

## 2. Data

目的は月々の顧客プロファイルに基づいて、将来のクレジットカードの支払い滞納を予測すること。ターゲットの二値は最新の明細から 18 ヵ月のパフォーマンスウィンドウを観察することによって計算され、そしてもし顧客が最新の明細から 120 日以内に支払いをしなかった場合、デフォルトイベントとみなす。
データセットは明細日付で各顧客ごとに集約されたプロファイル特徴を含む。特徴は匿名化標準化されそして、次の一般的なカテゴリーに落とし込まれる。

-   `D_*` =Delinquency variables 延滞変数
-   `S_*` =Spend variables 支出変数
-   `P_*` =Payment variables 支払変数
-   `B_*` =Balance variables バランス変数
-   `R_*` =Risk variables リスク変数

次の特徴はカテゴリーに分類される。
`['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']`
あなたの仕事はそれぞれの`customer_ID`について将来のデフォルトを`target=1`で予測すること。

-   `train_data.csv` <font color="red"><b>(16.39GB)</b></font>顧客ごとの明細データを持つ学習データ
-   `train_labels.csv` <font color="red">(30.75MB)</font>`customer_ID`ごとの`target`ラベル
-   `test_data.csv` <font color="red"><b>(33.82GB)</b></font>テストデータ;`customer_ID`ごとに`target`を予測する
-   `sample_submission.csv` <font color="red">(61.95MB)</font>正しい形式の提出サンプル

---

## 3. feather 形式に変換

[What is feather?](https://www.rstudio.com/blog/feather/)

> Feather: A Fast On-Disk Format for Data Frames for R and Python, powered by Apache Arrow

CSV データが重すぎて、このまま`read_csv`するとメモリが飛ぶ。そこで軽量な feather 形式に変換する。ついでに、データ型も換えておく。

@import "..\Development\convert_to_feather.py"

~~2 時間ぐらいかかった。~~

-   `train_data.csv` (16.39GB) -> ftr:**1.67GB**
-   `test_data.csv` (33.82GB) -> ftr:**3.44GB**

`pandas`で`to_feather`するには以下のライブラリが必要

```bash
Package           Version
----------------- -----------
pandas            1.4.2
pyarrow           8.0.0
```

---

## 4. データの概観

### 4.1. 明細数の確認

`train_data`の`customer_ID`毎に明細数をヒストグラム表示する。

<div align= "center">

<img src="src/target_histogram.svg" width=75%>

`顧客単位の明細データ数`

</div>

<div align= "center">

<img src="src/target_histogram_ex.svg" width=75%>

`拡大`

</div>

### 4.2. カテゴリ別に target を比較

`target`が`0`か`1`のそれぞれでランダムに 1 サンプルとり、パラメーターカテゴリー毎にプロットして比較する。

<div class="block_all">
<div class="block_left">

<div align= "center">

<img src="src/transition_Delinquency_0.svg">
<img src="src/transition_Spend_0.svg">
<img src="src/transition_Payment_0.svg">
<img src="src/transition_Balance_0.svg">
<img src="src/transition_Risk_0.svg">

`破産しない`

</div>

</div>

<div class="block_right">

<div align= "center">

<img src="src/transition_Delinquency_1.svg">
<img src="src/transition_Spend_1.svg">
<img src="src/transition_Payment_1.svg">
<img src="src/transition_Balance_1.svg">
<img src="src/transition_Risk_1.svg">

`破産する`

</div>

</div>
</div>

### 4.3. パラメーター毎に target で比較

`target`が`0`と`1`でそれぞれ 100 件分の`customer_ID`をサンプリングし、その明細データを各カテゴリー毎に比較する。

#### Delinquency variables

<div class="block_all">
<div class="block_left">

<div align= "center">

<img src="src/timescale_D_39.svg">
<img src="src/timescale_D_42.svg">
<img src="src/timescale_D_44.svg">
<img src="src/timescale_D_46.svg">
<img src="src/timescale_D_48.svg">
<img src="src/timescale_D_50.svg">
<img src="src/timescale_D_52.svg">
<img src="src/timescale_D_54.svg">
<img src="src/timescale_D_56.svg">
<img src="src/timescale_D_59.svg">
<img src="src/timescale_D_61.svg">
<img src="src/timescale_D_63.svg">
<img src="src/timescale_D_65.svg">
<img src="src/timescale_D_68.svg">
<img src="src/timescale_D_70.svg">
<img src="src/timescale_D_72.svg">
<img src="src/timescale_D_74.svg">
<img src="src/timescale_D_76.svg">
<img src="src/timescale_D_78.svg">
<img src="src/timescale_D_80.svg">
<img src="src/timescale_D_82.svg">
<img src="src/timescale_D_84.svg">
<img src="src/timescale_D_87.svg">
<img src="src/timescale_D_89.svg">
<img src="src/timescale_D_92.svg">
<img src="src/timescale_D_94.svg">
<img src="src/timescale_D_102.svg">
<img src="src/timescale_D_104.svg">
<img src="src/timescale_D_106.svg">
<img src="src/timescale_D_108.svg">
<img src="src/timescale_D_110.svg">
<img src="src/timescale_D_112.svg">
<img src="src/timescale_D_114.svg">
<img src="src/timescale_D_116.svg">
<img src="src/timescale_D_118.svg">
<img src="src/timescale_D_120.svg">
<img src="src/timescale_D_122.svg">
<img src="src/timescale_D_124.svg">
<img src="src/timescale_D_126.svg">
<img src="src/timescale_D_128.svg">
<img src="src/timescale_D_130.svg">
<img src="src/timescale_D_132.svg">
<img src="src/timescale_D_134.svg">
<img src="src/timescale_D_136.svg">
<img src="src/timescale_D_138.svg">
<img src="src/timescale_D_140.svg">
<img src="src/timescale_D_142.svg">
<img src="src/timescale_D_144.svg">
</div>
</div>

<div class="block_right">

<div align= "center">

<img src="src/timescale_D_41.svg">
<img src="src/timescale_D_43.svg">
<img src="src/timescale_D_45.svg">
<img src="src/timescale_D_47.svg">
<img src="src/timescale_D_49.svg">
<img src="src/timescale_D_51.svg">
<img src="src/timescale_D_53.svg">
<img src="src/timescale_D_55.svg">
<img src="src/timescale_D_58.svg">
<img src="src/timescale_D_60.svg">
<img src="src/timescale_D_62.svg">
<img src="src/timescale_D_64.svg">
<img src="src/timescale_D_66.svg">
<img src="src/timescale_D_69.svg">
<img src="src/timescale_D_71.svg">
<img src="src/timescale_D_73.svg">
<img src="src/timescale_D_75.svg">
<img src="src/timescale_D_77.svg">
<img src="src/timescale_D_79.svg">
<img src="src/timescale_D_81.svg">
<img src="src/timescale_D_83.svg">
<img src="src/timescale_D_86.svg">
<img src="src/timescale_D_88.svg">
<img src="src/timescale_D_91.svg">
<img src="src/timescale_D_93.svg">
<img src="src/timescale_D_96.svg">
<img src="src/timescale_D_103.svg">
<img src="src/timescale_D_105.svg">
<img src="src/timescale_D_107.svg">
<img src="src/timescale_D_109.svg">
<img src="src/timescale_D_111.svg">
<img src="src/timescale_D_113.svg">
<img src="src/timescale_D_115.svg">
<img src="src/timescale_D_117.svg">
<img src="src/timescale_D_119.svg">
<img src="src/timescale_D_121.svg">
<img src="src/timescale_D_123.svg">
<img src="src/timescale_D_125.svg">
<img src="src/timescale_D_127.svg">
<img src="src/timescale_D_129.svg">
<img src="src/timescale_D_131.svg">
<img src="src/timescale_D_133.svg">
<img src="src/timescale_D_135.svg">
<img src="src/timescale_D_137.svg">
<img src="src/timescale_D_139.svg">
<img src="src/timescale_D_141.svg">
<img src="src/timescale_D_143.svg">
<img src="src/timescale_D_145.svg">

</div>

</div>
</div>

---

#### Spend variables

<div class="block_all">
<div class="block_left">

<div align= "center">

<img src="src/timescale_S_3.svg">
<img src="src/timescale_S_6.svg">
<img src="src/timescale_S_9.svg">
<img src="src/timescale_S_12.svg">
<img src="src/timescale_S_15.svg">
<img src="src/timescale_S_17.svg">
<img src="src/timescale_S_19.svg">
<img src="src/timescale_S_22.svg">
<img src="src/timescale_S_24.svg">
<img src="src/timescale_S_26.svg">

</div>

</div>

<div class="block_right">

<div align= "center">

<img src="src/timescale_S_5.svg">
<img src="src/timescale_S_7.svg">
<img src="src/timescale_S_11.svg">
<img src="src/timescale_S_13.svg">
<img src="src/timescale_S_16.svg">
<img src="src/timescale_S_18.svg">
<img src="src/timescale_S_20.svg">
<img src="src/timescale_S_23.svg">
<img src="src/timescale_S_25.svg">
<img src="src/timescale_S_27.svg">

</div>

</div>
</div>

---

#### Payment variables

<div align= "center">

<img src="src/timescale_P_2.svg">
<img src="src/timescale_P_3.svg">
<img src="src/timescale_P_4.svg">

</div>

---

#### Balance variables

<div class="block_all">
<div class="block_left">

<div align= "center">

<img src="src/timescale_B_1.svg">
<img src="src/timescale_B_3.svg">
<img src="src/timescale_B_5.svg">
<img src="src/timescale_B_7.svg">
<img src="src/timescale_B_9.svg">
<img src="src/timescale_B_11.svg">
<img src="src/timescale_B_13.svg">
<img src="src/timescale_B_15.svg">
<img src="src/timescale_B_17.svg">
<img src="src/timescale_B_19.svg">
<img src="src/timescale_B_21.svg">
<img src="src/timescale_B_23.svg">
<img src="src/timescale_B_25.svg">
<img src="src/timescale_B_27.svg">
<img src="src/timescale_B_29.svg">
<img src="src/timescale_B_31.svg">
<img src="src/timescale_B_33.svg">
<img src="src/timescale_B_37.svg">
<img src="src/timescale_B_39.svg">
<img src="src/timescale_B_41.svg">

</div>

</div>

<div class="block_right">

<div align= "center">

<img src="src/timescale_B_2.svg">
<img src="src/timescale_B_4.svg">
<img src="src/timescale_B_6.svg">
<img src="src/timescale_B_8.svg">
<img src="src/timescale_B_10.svg">
<img src="src/timescale_B_12.svg">
<img src="src/timescale_B_14.svg">
<img src="src/timescale_B_16.svg">
<img src="src/timescale_B_18.svg">
<img src="src/timescale_B_20.svg">
<img src="src/timescale_B_22.svg">
<img src="src/timescale_B_24.svg">
<img src="src/timescale_B_26.svg">
<img src="src/timescale_B_28.svg">
<img src="src/timescale_B_30.svg">
<img src="src/timescale_B_32.svg">
<img src="src/timescale_B_36.svg">
<img src="src/timescale_B_38.svg">
<img src="src/timescale_B_40.svg">
<img src="src/timescale_B_42.svg">

</div>

</div>
</div>

---

#### Risk variables

<div class="block_all">
<div class="block_left">

<div align= "center">

<img src="src/timescale_R_1.svg">
<img src="src/timescale_R_3.svg">
<img src="src/timescale_R_5.svg">
<img src="src/timescale_R_7.svg">
<img src="src/timescale_R_9.svg">
<img src="src/timescale_R_11.svg">
<img src="src/timescale_R_13.svg">
<img src="src/timescale_R_15.svg">
<img src="src/timescale_R_17.svg">
<img src="src/timescale_R_19.svg">
<img src="src/timescale_R_21.svg">
<img src="src/timescale_R_23.svg">
<img src="src/timescale_R_25.svg">
<img src="src/timescale_R_27.svg">

</div>

</div>

<div class="block_right">

<div align= "center">

<img src="src/timescale_R_2.svg">
<img src="src/timescale_R_4.svg">
<img src="src/timescale_R_6.svg">
<img src="src/timescale_R_8.svg">
<img src="src/timescale_R_10.svg">
<img src="src/timescale_R_12.svg">
<img src="src/timescale_R_14.svg">
<img src="src/timescale_R_16.svg">
<img src="src/timescale_R_18.svg">
<img src="src/timescale_R_20.svg">
<img src="src/timescale_R_22.svg">
<img src="src/timescale_R_24.svg">
<img src="src/timescale_R_26.svg">
<img src="src/timescale_R_28.svg">

</div>

</div>
</div>

<style>
.block_all{width:800px;margin:0 auto;}
</style>
<style>
.block_left{width:400px;float:left;}
</style>
<style>
.block_right{width:400px;float:right;}
</style>

#### パラメーターの分類

-   カテゴリーデータ

    -   名義尺度

    ```
    ['D_63', 'D_64', (S_2)] S_2 は日付
    ```

    -   間隔尺度

    ```
    ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_66', 'D_68']
    ```

-   量的データ

    -   二値っぽいもの

    ```
    ['D_54', 'D_82', 'D_86', 'D_93', 'D_94', 'D_96', 'D_103', 'D_112',
    'D_123', 'D_127', 'D_128', 'D_129', 'D_130', 'D_140',
    'S_6', 'S_18', 'S_20',
    'B_8', 'B_31', 'B_32', 'B_33', 'B_41',
    'R_2', 'R_4', 'R_15', 'R_19', 'R_21', 'R_22', 'R_24', 'R_25', 'R_27']
    ```

    -   三値以上っぽいもの

    ```
    ['D_91', 'D_92', 'D_113', 'D_125', 'B_22']
    ```

-   使えなさそうなデータ
    ```
    ['D_66', 'D_69', 'D_73', 'D_87', 'D_88', 'D_108',
    'D_110', 'D_111', 'D_116', 'D_137', 'D_139', 'D_143',
    'B_39', 'B_42', 'R_28']
    ```

### 4.4. 相関行列と寄与率

#### Delinquency variables

<div align= "center">
<img src="src\correlation_matrix_D.svg">
<img src="src\contribution_ratio_D.svg">
</div>

---

#### Spend variables

<div align= "center">
<img src="src\correlation_matrix_S.svg">
<img src="src\contribution_ratio_S.svg">
</div>

---

#### Payment variables

<div align= "center">
<img src="src\correlation_matrix_P.svg">
<img src="src\contribution_ratio_P.svg">
</div>

---

#### Balance variables

<div align= "center">
<img src="src\correlation_matrix_B.svg">
<img src="src\contribution_ratio_B.svg">
</div>

---

#### Risk variables

<div align= "center">
<img src="src\correlation_matrix_R.svg">
<img src="src\contribution_ratio_R.svg">
</div>

---

## 5.1. LightGBM

https://pypi.org/project/lightgbm/

### GPU 版インストール手順

1. `Cmake`のインストール
   https://cmake.org/download/
2. `OpenCL`をインストール
   GPU のメーカーで分岐
    - Intel SDK for OpenCL
    - AMD APP SDK
    - NVIDIA CUDA Toolkit
      https://www.nvidia.co.jp/Download/index.aspx?lang=jp
3. ~~`Boost`のインストール~~
   https://sourceforge.net/projects/boost/postdownload
   から exe を落としてインストール
   警告がすごい出る

この手順いらない？`"device":"gpu"`でいけたっぽい

### dart

### 評価関数

```python
def amex_metric(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    # preds = y_pred[indices]
    target = y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)
```

## 6.1. 結果

### 顧客毎に最新の 2 明細で学習

-   単純なモデル
-   gbdt
-   カテゴリーパラメータは除外

スコア

```
0.785
```

### ハイパーパラメータチューニング

-   `optuna`でチューニング

スコア

```
0.783
```

### 主成分分析でパラメータを圧縮

-   パラメータを 5 つのカテゴリーに分け、それぞれで PCA モデルを作成、パラメータを圧縮
-   メモリの消費が激しく、5 つ全部は無理

スコア

```
0.768 # pca 80%
0.768 # pca 90%
```

### パラメータ毎に統計量を算出、カテゴリーパラメータは one-hot-encoding

-   統計量(`["mean", "std", "min", "max", "last"]`)を新たに算出
-   カテゴリーパラメータに one-hot-encoding を適用、`["sum", "last"]`を作る
-   パラメータ数が 900 近くになると、optuna がエラーを吐きまくるので、`feature_importance`が低いパラメータは学習から除外
-   boosting:dart
-   クロスバリデーション、`Kfold==3`

### 主成分分析で時間軸方向に圧縮

-   各パラメータは時系列データなので、時間軸を無次元化したい
