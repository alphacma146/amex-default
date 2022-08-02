# %%
# Standard libe
import gc
# Third party
import numpy as np
import pandas as pd
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb
from sklearn.model_selection import RepeatedKFold
# self made
from amex_base import \
    Config, \
    lgb_amex_metric,\
    lgb_crossvalid,\
    xy_set,\
    show_result,\
    save_predict

PARAM_SEARCH = False
ACTIVE_COL = 0.02

CFG = Config()
# %%
# 前処理


def preprocess(data: pd.DataFrame):
    # 時間データ
    MONTH = 30.5
    data["S_2"] = pd.to_datetime(data["S_2"])
    data.sort_values(["customer_ID", "S_2"], inplace=True)

    data["S_2_scale"] = (
        data[["customer_ID", "S_2"]]
        .groupby("customer_ID")
        .transform(lambda x: (x - x.iloc[0]) / pd.Timedelta(1, "D"))
    )
    data["S_2_scale"] = data["S_2_scale"].astype(np.int16)
    data["S_2_day"] = (
        data[["customer_ID", "S_2"]]
        .groupby("customer_ID")
        .transform(lambda x: x.dt.day)
    )
    data["S_2_day"] = data["S_2_day"].astype(np.int16)
    data["S_2_interval"] = (
        data[["customer_ID", "S_2"]]
        .groupby("customer_ID")
        .transform(
            lambda x: x.diff() / pd.Timedelta(1, "D")
        )
    )
    data["S_2_interval"].fillna(MONTH, inplace=True)
    data["S_2_interval"] = data["S_2_interval"].astype(np.float16)

    # データ数が少ないパラメータを削除
    if len(set(CFG.remove_param) & set(data.columns)) != 0:
        data.drop(CFG.remove_param, axis=1, inplace=True)

    # 時間の差分で補正
    ignore_col = CFG.category_param + ["customer_ID", "S_2_scale", "S_2_day"]
    num_col = [col for col in data.columns if col not in ignore_col]
    interval_data = data["S_2_interval"].copy()
    data.loc[:, num_col] = data[num_col].apply(
        lambda x: x * MONTH / x.S_2_interval, axis=1
    )
    data["S_2_interval"] = interval_data

    # パラメータを組み合わせて新たな特徴量を作成
    for c_col, p_col in [
        (c_col, p_col)
        for c_col in ["B_11", "B_14", "B_17", "D_39", "D_131", "S_16", "S_23"]
        for p_col in ["P_2", "P_3"]
    ]:
        data[f"{c_col}-{p_col}"] = data[c_col] - data[p_col]
        data[f"{c_col}/{p_col}"] = data[c_col] / data[p_col]

    # カテゴリーデータ->one hot encoding
    # 数値データ->集約データ
    cat_col = ["customer_ID"] + CFG.category_param
    num_col = [col for col in data.columns if col not in CFG.category_param]
    cat_data = data[cat_col]
    num_data = data[num_col]

    num_data = (
        num_data
        .groupby("customer_ID")
        .agg(["mean", "std", "min", "max", "last", "first"])
    )
    num_data.columns = ["_".join(col_list) for col_list in num_data.columns]

    agg_data = pd.DataFrame(index=num_data.index)
    agg_list = []
    for col in [col for col in num_col if col != "customer_ID"]:
        agg_frag = agg_data.copy()
        d_last = num_data[f"{col}_last"]
        d_first = num_data[f"{col}_first"]
        d_mean = num_data[f"{col}_mean"]
        d_max = num_data[f"{col}_max"]
        d_min = num_data[f"{col}_min"]
        agg_frag[f"{col}_last_first_div"] = d_last / d_first
        agg_frag[f"{col}_last_mean_sub"] = d_last - d_mean
        agg_frag[f"{col}_last_mean_div"] = d_last / d_mean
        agg_frag[f"{col}_last_max_div"] = d_last / d_max
        agg_frag[f"{col}_last_min_div"] = d_last / d_min
        agg_list.append(agg_frag)
    num_data = pd.concat([num_data] + agg_list, axis=1)

    cat_data = (
        pd.get_dummies(cat_data, columns=CFG.category_param)
        .groupby("customer_ID")
        .agg(["sum", "last", "first"])
    )
    cat_data.columns = ["_".join(col_list) for col_list in cat_data.columns]

    # testデータに含まれない値
    cat_data.drop(
        [col for col in ['D_68_0.0_sum', 'D_68_0.0_last',
                         'D_64_-1_sum', 'D_64_-1_last',
                         'D_66_0.0_sum', 'D_66_0.0_last']
         if col in cat_data.columns],
        axis=1,
        inplace=True
    )

    gc.collect()

    return pd.merge(num_data, cat_data, left_index=True, right_index=True)


train_data = preprocess(pd.read_feather(CFG.train_data_path))
train_labels = (
    pd.read_csv(CFG.train_label_path)
    .set_index('customer_ID', drop=True)
    .sort_index()
)
# %%
# 重要度が低いパラメータを学習から除外
train_labels = pd.merge(
    train_data[[]],
    train_labels,
    how="left",
    left_index=True,
    right_index=True
)
(
    train_set,
    valid_set,
    x_valid,
    y_valid
) = xy_set(train_data, train_labels)
model = lgb.train(
    CFG.model_param,
    train_set=train_set,
    valid_sets=valid_set,
    feval=lgb_amex_metric,
    num_boost_round=1000,
    callbacks=[
        # lgb.early_stopping(100),
        lgb.log_evaluation(0),
    ]
)
importance = pd.DataFrame(
    data={
        "col": x_valid.columns,
        "importance": model.feature_importance()
    }
).sort_values(["importance"])
importance["ratio"] = (
    importance["importance"]
    / importance["importance"].sum()
    * 100
)
use_col = importance.query(f"ratio>={ACTIVE_COL}")["col"]
print(use_col)
# %%
# パラメータチューニング
train_set = lgb.Dataset(train_data, train_labels)
match PARAM_SEARCH:
    case True:
        params = CFG.model_param
        params["force_col_wise"] = True
        params["n_jobs"] = 8
        tuner = opt_lgb.LightGBMTunerCV(
            params,
            train_set=train_set,
            feval=lgb_amex_metric,
            num_boost_round=500,
            folds=RepeatedKFold(n_splits=3, n_repeats=1, random_state=37),
            shuffle=True,
            callbacks=[
                # lgb.early_stopping(50),
                lgb.log_evaluation(50),
            ]
        )
        tuner.run()
        params = tuner.best_params
    case False:
        params = {
            'colsample_bytree': 1.0,
            'importance_type': 'split',
            'min_child_samples': 100,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0,
            'n_estimators': 100,
            'n_jobs': 8,
            'num_leaves': 255,
            'random_state': None,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'silent': 'warn',
            'subsample': 1.0,
            'subsample_for_bin': 200000,
            'subsample_freq': 0,
            'force_col_wise': True,
            'feature_pre_filter': False,
            'lambda_l1': 3.0542906593883284e-06,
            'lambda_l2': 0.6233074695450682,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
        }
        # 0.7883137925427842

params |= CFG.model_param
# cross validation
cv_score = lgb_crossvalid(train_data[use_col], train_labels["target"], params)
print(f"Amex CVScore: {np.mean(cv_score)}")
# create model
model = lgb.LGBMClassifier(**params, num_boost_round=1000)
model.fit(
    train_data,
    train_labels["target"],
    callbacks=[
        # lgb.early_stopping(100),
        lgb.log_evaluation(50),
    ]
)
# visualize
show_result(model, train_data, train_labels["target"])
# %%
# predict
test_data = preprocess(pd.read_feather(CFG.test_data_path))
save_predict(
    model,
    test_data[use_col],
    CFG.sample_submission_path,
    CFG.result_folder_path / "res_sub_agg.csv"
)
# %%
