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
from sklearn import metrics, model_selection
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

PARAM_SEARCH = False


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
        "max_bin": 255,
        "max_depth": -1,
        # "bagging_fraction": 0.7300867687463426,
        "feature_fraction": 0.545038803605222
    })


CFG = Config()
# {
# 'device': 'gpu',
# 'objective': 'binary',
# 'max_bin': 255,
# 'max_depth': -1,
# 'feature_fraction': 0.4,
# 'feature_pre_filter': False,
# 'lambda_l1': 0.38700098861246723,
# 'lambda_l2': 0.270693422083792,
# 'num_leaves': 253,
# 'bagging_fraction': 1.0,
# 'bagging_freq': 0,
# 'min_child_samples': 100,
# 'num_iterations': 300,
# 'early_stopping_round': None
# }
# 0.9045479944341576

# %%
# 最新の明細で学習する
train_data = (
    pd.read_feather(CFG.train_data_path)
    .groupby('customer_ID')
    .tail(2)
    .set_index('customer_ID', drop=True)
    .sort_index()
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
remove = ["S_2", "D_63", "D_64"]

train, valid = model_selection.train_test_split(
    train_data.drop(remove, axis=1),
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
            num_boost_round=300,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(0),
            ]
        )
        param = opt_model.params
    case False:
        param = {
            'feature_fraction': 0.4,
            'feature_pre_filter': False,
            'lambda_l1': 0.38700098861246723,
            'lambda_l2': 0.270693422083792,
            'num_leaves': 253,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_child_samples': 100,
            'num_iterations': 300,
        }


end_time = time.time()
# %%
model = lgb.train(
    CFG.model_param | param,
    train_set=train_set,
    valid_sets=valid_set,
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(0),
    ]
)
# %%
result = model.predict(valid.drop("target", axis=1))
score = metrics.accuracy_score(
    valid["target"],
    np.where(result < 0.5, 0, 1)
)
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
test_data = pd.read_feather(CFG.test_data_path)
test_data["date"] = pd.to_datetime(test_data["S_2"], format="%Y-%m-%d")
test_data = (
    test_data
    .loc[
        test_data
        .groupby("customer_ID")["date"]
        .idxmax()
    ]
    .set_index("customer_ID", drop=True)
)

sub = pd.read_csv(CFG.sample_submission_path)
sub["prediction"] = model.predict(test_data.drop(["date", *remove], axis=1))
sub.to_csv(CFG.result_submission_path, index=False)
# %%
