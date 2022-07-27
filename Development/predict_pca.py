# %%
# Standard lib
import pickle
# Third party
import numpy as np
import pandas as pd
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from tqdm import tqdm
# self made
from amex_base import \
    Config, \
    lgb_amex_metric,\
    xy_set,\
    show_result,\
    save_predict

USE_PICKLE = True
PARAM_SEARCH = False
N_COMP = 2

CFG = Config()
# %%


def preprocess(data: pd.DataFrame) -> pd.DataFrame:

    data.sort_values(["customer_ID", "S_2"], inplace=True)
    data.fillna(0, inplace=True)

    if len(set(CFG.remove_param) & set(data.columns)) != 0:
        data = data.drop(CFG.remove_param, axis=1)

    data = pd.get_dummies(data, columns=CFG.category_param)
    data.drop(
        [col for col in ['D_68_0.0', 'D_64_-1', 'D_66_0.0']
         if col in data.columns],
        axis=1,
        inplace=True
    )

    return (
        data
        .set_index("customer_ID", drop=True)
    )


train_data = preprocess(pd.read_feather(CFG.train_data_path))
train_labels = (
    pd.read_csv(CFG.train_label_path)
    .set_index('customer_ID', drop=True)
    .sort_index()
)
# %%


def transpose_data(data: pd.DataFrame) -> pd.DataFrame:

    res_list = []

    for cid, df in data.groupby("customer_ID").__iter__():
        df_val = list(df.values)
        if (df_len := len(df_val)) != 13:
            df_val = [0] * (13 - df_len) + df_val
        res_list.append([cid] + df_val)

    return pd.DataFrame(
        data=res_list,
        columns=np.concatenate(
            [["id"], [f"series_{i}" for i in np.arange(13)]]
        )
    ).set_index("id", drop=True)


def create_model(data: pd.DataFrame) -> tuple[PCA, np.array]:

    ss = StandardScaler()
    pca = PCA(n_components=N_COMP, svd_solver="full")

    pca.fit(ss.fit_transform(data))

    return pca, np.cumsum(pca.explained_variance_ratio_)


match USE_PICKLE:
    case True:
        with open("model_score.pickle", mode="rb") as f:
            model_dict = pickle.load(f)
            score_dict = pickle.load(f)
    case False:
        model_dict = {}
        score_dict = {}
        for col in tqdm(train_data.columns):
            t_data = transpose_data(train_data[col])
            model_dict[col], score_dict[col] = create_model(t_data)

        with open("model_score.pickle", mode="wb") as f:
            pickle.dump(model_dict, f)
            pickle.dump(score_dict, f)

score_df = pd.DataFrame(
    [x[-1] for x in score_dict.values()], index=score_dict.keys()
)
fig = px.bar(score_df)
fig.show()
# %%


def transform_data(data: pd.DataFrame) -> dict["col":np.array]:
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
    index=train_labels.index
)
(
    train_set,
    valid_set,
    x_valid,
    y_valid
) = xy_set(train_data, train_labels["target"])

match PARAM_SEARCH:
    case True:
        tuner = opt_lgb.LightGBMTunerCV(
            CFG.model_param,
            train_set=train_set,
            feval=lgb_amex_metric,
            num_boost_round=500,
            folds=RepeatedKFold(n_splits=3, n_repeats=3, random_state=37),
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
            'feature_pre_filter': False,
            'lambda_l1': 1.7896583416748754e-08,
            'lambda_l2': 2.1012022016175806,
            'num_leaves': 250,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8851939418050575,
            'bagging_freq': 1,
            'min_child_samples': 100,
        }
        # 0.76831612669674
# %%
params |= CFG.model_param
(
    train_set,
    x_valid,
    valid_set,
    y_valid
) = model_selection.train_test_split(
    train_data, train_labels["target"],
    test_size=0.2,
    random_state=0
)
model = lgb.LGBMClassifier(**params, num_boost_round=1000)
model.fit(
    train_set,
    valid_set,
    eval_set=[(x_valid, y_valid)],
    # eval_metric=lgb_amex_metric,
    callbacks=[
        # lgb.early_stopping(100),
        lgb.log_evaluation(50),
    ]
)
show_result(model, x_valid, y_valid)
# %%
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
