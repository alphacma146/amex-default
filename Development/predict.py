# %%
# Standard lib
import time
from pathlib import Path
from dataclasses import dataclass, field
# Third party
import numpy as np
import pandas as pd
import optuna.integration.lightgbm as opt_lgb
import lightgbm as lgb
from sklearn import model_selection
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

DATA_TYPE = "pca"  # pca or normal
PARAM_SEARCH = True


@dataclass
class Config():
    train_data_path: Path = Path(r"Data\feather_data\train_data.ftr")
    test_data_path: Path = Path(r"Data\feather_data\test_data.ftr")
    train_pca_data_path: Path = Path(r"Data\feather_data\train_pca_data.ftr")
    test_pca_data_path: Path = Path(r"Data\feather_data\test_pca_data.ftr")
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
        "max_bin": 255,
        "max_depth": -1,
        # "bagging_fraction": 0.7300867687463426,
        # "feature_fraction": 0.545038803605222
    })
    remove_parameter: list = field(default_factory=lambda: [
        "S_2", "D_63", "D_64"
    ])


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
# 最新の明細で学習する
match DATA_TYPE:
    case "normal":
        train_path = CFG.train_data_path
        test_path = CFG.test_data_path
        remove_list = CFG.remove_parameter
    case "pca":
        train_path = CFG.train_pca_data_path
        test_path = CFG.test_pca_data_path
        remove_list = []

train_data = (
    pd.read_feather(train_path)
    .groupby('customer_ID')
    .tail(2)
    .set_index('customer_ID', drop=True)
    .sort_index()
    .drop(remove_list, axis=1)
)
train_labels = (
    pd.read_csv(CFG.train_label_path)
    .set_index('customer_ID', drop=True)
    .sort_index()
)
train_data = pd.merge(
    train_data,
    train_labels,
    left_index=True,
    right_index=True
)
# %%
# パラメータチューニング
start_time = time.time()

train, valid = model_selection.train_test_split(
    train_data,
    test_size=0.2,
    random_state=0
)
train_set = lgb.Dataset(
    train.drop("target", axis=1),
    train["target"]
)
valid_set = lgb.Dataset(
    valid.drop("target", axis=1),
    valid["target"],
    reference=train_set
)

match PARAM_SEARCH:
    case True:
        opt_model = opt_lgb.train(
            CFG.model_param,
            train_set=train_set,
            valid_sets=valid_set,
            feval=lgb_amex_metric,
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0),
            ]
        )
        param = opt_model.params
    case False:
        match DATA_TYPE:
            case "normal":
                param = {
                    'feature_pre_filter': False,
                    'lambda_l1': 0.3916093444694036,
                    'lambda_l2': 0.010040374004305474,
                    'num_leaves': 241,
                    'feature_fraction': 0.4,
                    'bagging_fraction': 1.0,
                    'bagging_freq': 0,
                    'min_child_samples': 10,
                    'num_iterations': 500,
                }
                # 0.7930567263600699
            case "pca":
                param = {
                    'feature_fraction': 0.748,
                    'feature_pre_filter': False,
                    'lambda_l1': 1.8813865762117634,
                    'lambda_l2': 1.9355417225259987e-05,
                    'num_leaves': 256,
                    'bagging_fraction': 1.0,
                    'bagging_freq': 0,
                    'min_child_samples': 20,
                    'num_iterations': 300,
                    # 0.8994916238454712
                }


end_time = time.time()
# %%
model = lgb.train(
    CFG.model_param | param,
    train_set=train_set,
    valid_sets=valid_set,
    feval=lgb_amex_metric,
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(0),
    ]
)
# %%
result = model.predict(valid.drop("target", axis=1))
score = amex_metric(valid["target"], result)
print(model.params, score, sep="\n")
# %%
# output
sns.displot(result, bins=20, kde=True, rug=False)
plt.show()
lgb.plot_importance(model)
plt.show()

print(
    f"{end_time-start_time:.3f}".ljust(7, "_")
    + f": {score:.3f}"
    + "[-]"
)
# %%
# predict
test_data = (
    pd.read_feather(test_path)
    .groupby('customer_ID')
    .tail(1)
    .set_index('customer_ID', drop=True)
    .sort_index()
    .drop(remove_list, axis=1)
)

sub = pd.read_csv(CFG.sample_submission_path)
sub["prediction"] = model.predict(test_data)
sub.to_csv(CFG.result_submission_path, index=False)
# %%
