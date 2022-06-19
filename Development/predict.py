# %%
# Standard lib
import time
from pathlib import Path
from dataclasses import dataclass, field
# Third party
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics, model_selection
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


@dataclass
class Config():
    train_data_path: Path = Path(r"Data\feather_data\train_data.ftr")
    test_data_path: Path = Path(r"Data\feather_data\test_data.ftr")
    train_label_path: Path = Path(
        r"Data\amex-default-prediction\train_labels.csv"
    )
    result_submission_path: Path = Path(r"Data\result_submission.csv")
    model_param: dict = field(default_factory=lambda: {
        "task": "train",
        "boosting": "gbdt",
        "objective": "binary",
        "metric": {"binary_logloss"},
        'num_leaves': 13,
        'min_data_in_leaf': 7,
        'max_depth': -1,
        "force_col_wise": True,
        "device": "gpu"
    })


CFG = Config()


# %%
# 最新の明細で学習する
train_data = pd.read_feather(CFG.train_data_path)
train_label = pd.read_csv(CFG.train_label_path)
train_data["date"] = pd.to_datetime(train_data["S_2"], format="%Y-%m-%d")
latest_statement = (
    train_data
    .loc[
        train_data
        .groupby("customer_ID")["date"]
        .idxmax()
    ]
)
latest_statement = pd.merge(
    latest_statement,
    train_label,
    left_on="customer_ID",
    right_on="customer_ID"
)
latest_statement[["customer_ID", "date", "target"]].head()
# %%
# 学習
start_time = time.time()
res_list = []
remove = ["customer_ID", "S_2", "date", "D_63", "D_64"]
for _ in range(30):
    train, valid = model_selection.train_test_split(
        latest_statement.drop(remove, axis=1), test_size=0.1
    )
    train_lgb = lgb.Dataset(
        train.drop("target", axis=1),
        train["target"]
    )
    valid_lgb = lgb.Dataset(
        valid.drop("target", axis=1),
        valid["target"]
    )
    model_lgb = lgb.train(
        CFG.model_param,
        train_lgb,
        valid_sets=valid_lgb,
        num_boost_round=10000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(0),
        ]
    )
    result = model_lgb.predict(valid.drop("target", axis=1))
    score = metrics.accuracy_score(
        valid["target"],
        np.where(result < 0.5, 0, 1)
    )
    res_list.append((model_lgb, result, score))
    print(score)
end_time = time.time()
# %%
# output
best_model, best_result, best_score = max(
    res_list, key=lambda x: x[2]
)
sns.displot(best_result, bins=20, kde=True, rug=False)
plt.show()
lgb.plot_importance(best_model)
plt.show()
print(best_score)

print(
    f"{start_time-end_time:.3f}".ljust(7, "_")
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
)
result = pd.DataFrame(
    data={
        "prediction": np.where(
            best_model.predict(test_data.drop(remove, axis=1)) < 0.5,
            0, 1
        )
    },
    index=test_data["customer_ID"]
)
result.to_csv(CFG.result_submission_path)
# %%
