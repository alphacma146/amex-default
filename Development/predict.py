# %%
# Standard libe
from pathlib import Path
from dataclasses import dataclass, field
# Third party
import numpy as np
import pandas as pd
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb
from sklearn import model_selection
from sklearn.model_selection import KFold
import plotly.express as px

PARAM_SEARCH = True


@dataclass
class Config():
    train_data_path: Path = Path(r"Data\feather_data\train_data.ftr")
    test_data_path: Path = Path(r"Data\feather_data\test_data.ftr")
    train_label_path: Path = Path(
        r"Data\amex-default-prediction\train_labels.csv"
    )
    sample_submission_path: Path = Path(
        r"Data\amex-default-prediction\sample_submission.csv"
    )
    result_submission_path: Path = Path(r"Data\result_submission.csv")
    model_param: dict = field(default_factory=lambda: {
        "device": "gpu",
        "objective": "binary",
        "boosting": "dart",
        "verbose": -1,
        "max_bins": 250,
        "learning_rate": 0.05,
    })
    category_param: list = field(
        default_factory=lambda: [
            'D_63', 'D_64',  # categorical
            'B_30', 'B_38',
            'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_66', 'D_68'
        ]
    )
    remove_param: list = field(
        default_factory=lambda: [
            "S_2",
            "D_73", "D_87", "D_88", "D_108", "D_110", "D_111", "B_39", "B_42"
        ]
    )


CFG = Config()


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


def lgb_amex_metric(y_pred, y_true):
    return (
        "Score",
        amex_metric(y_true.get_label(), y_pred),
        True
    )


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
        .fillna(0)
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


def xy_set(x_data: pd.DataFrame, y_data: pd.DataFrame):
    x_t, x_v, y_t, y_v = model_selection.train_test_split(
        x_data, y_data,
        test_size=0.2,
        random_state=0
    )
    t_set = lgb.Dataset(x_t, y_t)
    v_set = lgb.Dataset(x_v, y_v, reference=t_set)

    return t_set, v_set, x_v, y_v


def show_result(model, x_value, y_value):
    result = model.predict(x_value)
    score = amex_metric(y_value, result)
    print(model.params, score, sep="\n")
    fig = px.histogram(result, nbins=100)
    fig.show()
    importance = pd.DataFrame(
        data={
            "importance": model.feature_importance(),
            "col": train_data.columns
        }
    ).sort_values(["importance"])
    fig = px.bar(importance["importance"])
    fig.show()

    return importance


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
use_col = res_df.query("ratio>=0.1")["col"]
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
            'lambda_l1': 0.3916093444694036,
            'lambda_l2': 0.010040374004305474,
            'num_leaves': 241,
            'feature_fraction': 0.4,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_child_samples': 10
        }
        # 0.7930567263600699
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
# %%
# predict
test_data = preprocess(pd.read_feather(CFG.test_data_path))
res_df = pd.DataFrame(
    data={"prediction": model.predict(test_data[use_col])},
    index=test_data.index
)
sub_df = pd.read_csv(CFG.sample_submission_path, usecols=["customer_ID"])
sub_df = sub_df.merge(res_df, left_on="customer_ID", right_index=True)
print(sub_df.head(10))
sub_df.to_csv(CFG.result_submission_path, index=False)
# %%
