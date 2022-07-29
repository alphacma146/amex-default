# %%
# Standard libe
# Third party
import numpy as np
import pandas as pd
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
# self made
from amex_base import \
    Config, \
    lgb_amex_metric,\
    xy_set,\
    show_result,\
    save_predict

PARAM_SEARCH = True
ACTIVE_COL = 0.02

CFG = Config()
# %%


def preprocess(data: pd.DataFrame):
    data["S_2"] = pd.to_datetime(data["S_2"])
    data.sort_values(["customer_ID", "S_2"], inplace=True)

    data["S_2_scale"] = (
        data[["customer_ID", "S_2"]]
        .groupby("customer_ID")
        .transform(lambda x: (x - x.iloc[0]) / pd.Timedelta(1, "D"))
    )
    data["S_2_interval"] = (
        data[["customer_ID", "S_2"]]
        .groupby("customer_ID")
        .transform(
            lambda x: [0] + (np.diff(x) / pd.Timedelta(1, "D")).tolist()
        )
    )

    if len(set(CFG.remove_param) & set(data.columns)) != 0:
        data.drop(CFG.remove_param, axis=1, inplace=True)

    cat_data = data[["customer_ID"] + CFG.category_param]
    num_data = data[
        [col for col in data.columns if col not in CFG.category_param]
    ]
    num_data = (
        num_data
        .groupby("customer_ID")
        .agg(["mean", "std", "min", "max", "last"])
    )
    num_data.columns = ["_".join(col_list) for col_list in num_data.columns]
    cat_data = (
        pd.get_dummies(cat_data, columns=CFG.category_param)
        .groupby("customer_ID")
        .agg(["sum", "last"])
    )
    cat_data.columns = ["_".join(col_list) for col_list in cat_data.columns]
    cat_data.drop(
        [col for col in ['D_68_0.0_sum', 'D_68_0.0_last',
                         'D_64_-1_sum', 'D_64_-1_last',
                         'D_66_0.0_sum', 'D_66_0.0_last']
         if col in cat_data.columns],
        axis=1,
        inplace=True
    )

    return pd.merge(num_data, cat_data, left_index=True, right_index=True)


train_data = preprocess(pd.read_feather(CFG.train_data_path))
train_labels = (
    pd.read_csv(CFG.train_label_path)
    .set_index('customer_ID', drop=True)
    .sort_index()
)
# %%
(
    train_set,
    valid_set,
    x_valid,
    y_valid
) = xy_set(train_data, train_labels["target"])
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
(
    train_set,
    valid_set,
    x_valid,
    y_valid
) = xy_set(train_data[use_col], train_labels["target"])
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
            folds=RepeatedKFold(n_splits=3, n_repeats=3, random_state=37),
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
        # 0.7862325624171822

params |= CFG.model_param
(
    train_set,
    x_valid,
    valid_set,
    y_valid
) = model_selection.train_test_split(
    train_data[use_col], train_labels["target"],
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
# predict
test_data = preprocess(pd.read_feather(CFG.test_data_path))
save_predict(
    model,
    test_data[use_col],
    CFG.sample_submission_path,
    CFG.result_folder_path / "res_sub_agg.csv"
)
# %%
