# %%
"""
移動平均を算出
時間軸方向にPCAで圧縮
"""
# Standard lib
import pickle
# Third party
import numpy as np
import pandas as pd
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from tqdm import tqdm
# self made
from amex_base import \
    Config, \
    lgb_amex_metric,\
    lgb_crossvalid,\
    show_result,\
    save_predict

USE_PICKLE = True
PARAM_SEARCH = False
N_COMP = 2

CFG = Config()
# %%
# 前処理


def preprocess(data: pd.DataFrame) -> pd.DataFrame:

    data.sort_values(["customer_ID", "S_2"], inplace=True)

    if len(set(CFG.remove_param) & set(data.columns)) != 0:
        data = data.drop(CFG.remove_param, axis=1)

    # 移動平均
    for col in [
        col for col in data.columns
        if col not in ["customer_ID"] + CFG.category_param
    ]:
        data[col] = (
            data[["customer_ID", col]]
            .groupby("customer_ID", as_index=False)
            .rolling(3)
            .mean()
        )[col]
        data[col].astype(np.float16)

    # 最新2明細は0なので除外
    data = (
        data
        .groupby("customer_ID", as_index=False)
        .tail(-2)
    )

    # one hot
    data = pd.get_dummies(data, columns=CFG.category_param)

    data.set_index("customer_ID", inplace=True)
    data.drop(
        [col for col in ['D_68_0.0', 'D_64_-1', 'D_66_0.0']
         if col in data.columns],
        axis=1,
        inplace=True
    )
    data.fillna(0, inplace=True)

    return data


train_data = preprocess(pd.read_feather(CFG.train_data_path))
train_labels = (
    pd.read_csv(CFG.train_label_path)
    .set_index('customer_ID', drop=True)
    .sort_index()
)
# %%
# create PCA model and save pickle


def transpose_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    顧客別に明細を転置
    """

    ST_NUM = 11
    res_list = []

    for cid, df in data.groupby("customer_ID").__iter__():
        df_val = list(df.values)
        if (df_len := len(df_val)) != ST_NUM:
            df_val = [0] * (ST_NUM - df_len) + df_val
        res_list.append([cid] + df_val)

    return pd.DataFrame(
        data=res_list,
        columns=np.concatenate(
            [["id"], [f"series_{i}" for i in np.arange(ST_NUM)]]
        )
    ).set_index("id", drop=True)


def create_model(data: pd.DataFrame) -> tuple[PCA, np.array]:
    """
    PCA model
    """

    ss = StandardScaler()
    pca = PCA(n_components=N_COMP, svd_solver="full")

    pca.fit(ss.fit_transform(data))

    return pca, np.cumsum(pca.explained_variance_ratio_)


match USE_PICKLE:
    case True:
        with open("model_score.pkl", mode="rb") as f:
            model_dict = pickle.load(f)
            score_dict = pickle.load(f)
    case False:
        model_dict = {}
        score_dict = {}
        for col in tqdm(train_data.columns):
            t_data = transpose_data(train_data[col])
            model_dict[col], score_dict[col] = create_model(t_data)

        with open("model_score.pkl", mode="wb") as f:
            pickle.dump(model_dict, f)
            pickle.dump(score_dict, f)

score_df = pd.DataFrame(
    [x for _, x in score_dict.values()], index=score_dict.keys()
)
fig = px.bar(score_df)
fig.show()
# %%
# パラメータチューニング


def transform_data(data: pd.DataFrame) -> dict["col":np.array]:
    """
    圧縮して変換
    """
    comp_dict = {}
    ss = StandardScaler()
    for col in tqdm(data.columns):
        res = model_dict[col].transform(
            ss.fit_transform(transpose_data(data[col]))
        )
        for i, val in enumerate(res.transpose()):
            comp_dict[f"PA_{col}_{i}"] = val

    return comp_dict


train_data = pd.DataFrame(
    data=transform_data(train_data),
    index=train_data.index.unique()
)
train_labels = pd.merge(
    train_data[[]],
    train_labels,
    how="left",
    left_index=True,
    right_index=True
)
train_set = lgb.Dataset(train_data, train_labels["target"])

match PARAM_SEARCH:
    case True:
        tuner = opt_lgb.LightGBMTunerCV(
            CFG.model_param,
            train_set=train_set,
            feval=lgb_amex_metric,
            num_boost_round=500,
            folds=RepeatedKFold(n_splits=3, n_repeats=1, random_state=37),
            shuffle=True,
            callbacks=[
                # lgb.early_stopping(50),
                lgb.log_evaluation(0),
            ]
        )
        tuner.run()
        params = tuner.best_params
    case False:
        params = {
            'class_weight': None,
            'colsample_bytree': 1.0,
            'importance_type': 'split',
            'min_child_samples': 50,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0,
            'n_estimators': 100,
            'num_leaves': 243,
            'random_state': None,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'silent': 'warn',
            'subsample': 1.0,
            'subsample_for_bin': 200000,
            'subsample_freq': 0,
            'feature_pre_filter': False,
            'lambda_l1': 1.2958112509809807e-07,
            'lambda_l2': 0.3594257455906693,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
        }
        # Amex CVScore: 0.7659090803590255
# %%
# create model
params |= CFG.model_param
cv_score = lgb_crossvalid(train_data, train_labels["target"], params)
print(f"Amex CVScore: {np.mean(cv_score)}")

model = lgb.LGBMClassifier(**params, num_boost_round=1000)
model.fit(
    train_data,
    train_labels["target"],
    callbacks=[
        # lgb.early_stopping(100),
        lgb.log_evaluation(200),
    ]
)
show_result(model, train_data, train_labels["target"])
# %%
# predict
del train_data
test_data = preprocess(pd.read_feather(CFG.test_data_path))
test_data = pd.DataFrame(
    data=transform_data(test_data),
    index=test_data.index.unique()
)
save_predict(
    model,
    test_data,
    CFG.sample_submission_path,
    CFG.result_folder_path / "res_sub_pca.csv"
)
# %%
