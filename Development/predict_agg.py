# %%
# Standard libe
# Third party
import pandas as pd
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb
from sklearn.model_selection import KFold
# self made
from amex_base import \
    Config, \
    lgb_amex_metric,\
    xy_set,\
    show_result,\
    save_predict

PARAM_SEARCH = False

CFG = Config()
# %%


def preprocess(data: pd.DataFrame):
    data.sort_values(["customer_ID", "S_2"], inplace=True)

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
res_df = show_result(model, x_valid, y_valid)
res_df["ratio"] = (
    res_df["importance"]
    / res_df["importance"].sum()
    * 100
)
use_col = res_df.query("ratio>=0.05")["col"]
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
        tuner = opt_lgb.LightGBMTunerCV(
            CFG.model_param,
            train_set=train_set,
            feval=lgb_amex_metric,
            num_boost_round=500,
            folds=KFold(n_splits=3),
            callbacks=[
                # lgb.early_stopping(50),
                lgb.log_evaluation(0),
            ]
        )
        tuner.run()
        param = tuner.best_params
    case False:
        param = {
            'feature_pre_filter': False,
            'lambda_l1': 0.583389841517446,
            'lambda_l2': 5.880437992520181,
            'num_leaves': 214,
            'feature_fraction': 1.0,
            'bagging_fraction': 0.6810221344608107,
            'bagging_freq': 1,
            'min_child_samples': 50
        }
        # .7852266832010086
# %%
model = lgb.train(
    CFG.model_param | param,
    train_set=train_set,
    valid_sets=valid_set,
    feval=lgb_amex_metric,
    num_boost_round=1000,
    callbacks=[
        # lgb.early_stopping(100),
        lgb.log_evaluation(0),
    ]
)
show_result(model, x_valid, y_valid)
# predict
test_data = preprocess(pd.read_feather(CFG.test_data_path))
save_predict(
    model,
    test_data[use_col],
    CFG.sample_submission_path,
    CFG.result_folder_path / "res_sub_agg.csv"
)
# %%
