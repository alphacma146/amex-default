# %%
"""
最新の3明細で学習
"""
# Standard lib
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
    show_result,\
    save_predict

PARAM_SEARCH = False
NUM_STATEMENT = 3

CFG = Config()
# %%
# 前処理


def preprocess(data: pd.DataFrame) -> pd.DataFrame:

    # 最新3明細を取得
    # one hot encoding
    data = pd.get_dummies(
        (data
         .groupby("customer_ID")
         .tail(NUM_STATEMENT)),
        columns=CFG.category_param
    )
    data.drop(
        [col for col in ['D_68_0.0', 'D_64_-1', 'D_66_0.0']
         if col in data.columns],
        axis=1,
        inplace=True
    )
    data.sort_values(["customer_ID", "S_2"], inplace=True)
    data.drop(["S_2"], axis=1, inplace=True)

    return data.set_index('customer_ID', drop=True)


train_data = preprocess(pd.read_feather(CFG.train_data_path))
train_labels = (
    pd.read_csv(CFG.train_label_path)
    .set_index('customer_ID', drop=True)
    .sort_index()
)
train_labels = pd.merge(
    train_data[[]],
    train_labels,
    how="left",
    left_index=True,
    right_index=True
)
# %%
# パラメータチューニング
train_set = lgb.Dataset(train_data, train_labels["target"])
match PARAM_SEARCH:
    case True:
        params = CFG.model_param
        params["force_row_wise"] = True
        tuner = opt_lgb.LightGBMTunerCV(
            params,
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
            'force_row_wise': True,
            'feature_pre_filter': False,
            'lambda_l1': 0.7575741231595549,
            'lambda_l2': 0.006559531182232793,
            'num_leaves': 255,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_child_samples': 5
        }
        # 0.8102238689267146
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
        lgb.log_evaluation(50),
    ]
)
show_result(model, train_data, train_labels["target"])
# %%
# predict
test_data = (
    preprocess(pd.read_feather(CFG.test_data_path))
    .groupby("customer_ID")
    .tail(1)
)
save_predict(
    model,
    test_data,
    CFG.sample_submission_path,
    CFG.result_folder_path / "res_sub_lst.csv",
    func=lambda x: (
        x
        .groupby(level=0)
        .agg({"prediction": lambda x: np.mean(x)})
    )
)
# %%
