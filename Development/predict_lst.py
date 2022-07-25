# %%
# Standard libe
# Third party
import numpy as np
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
NUM_STATEMENT = 3

CFG = Config()
# %%


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data.sort_values(["customer_ID", "S_2"], inplace=True)

    if len(set(CFG.remove_param) & set(data.columns)) != 0:
        data.drop(CFG.remove_param, axis=1, inplace=True)

    data = pd.get_dummies(
        (data
         .groupby("customer_ID")
         .tail(NUM_STATEMENT)),
        columns=CFG.category_param
    )

    data.drop(
        [col for col in ['D_68_0.0_sum', 'D_68_0.0_last',
                         'D_64_-1_sum', 'D_64_-1_last',
                         'D_66_0.0_sum', 'D_66_0.0_last']
         if col in data.columns],
        axis=1,
        inplace=True
    )

    return data


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
    test_data,
    CFG.sample_submission_path,
    CFG.result_folder_path / "res_sub_lst.csv",
    func=lambda x: (
        x
        .groupby(level=0)
        .apply(lambda x: np.power(np.cumprod(x), 1 / len(x)))
    )
)
# %%
