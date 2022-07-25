# Standard libe
from pathlib import Path
from dataclasses import dataclass, field
# Third party
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import model_selection
import plotly.express as px


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
    result_folder_path: Path = Path(r"Data\result")
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


def lgb_amex_metric(y_pred, y_true) -> tuple:
    """
    lgb wrapper
    """
    return (
        "Score",
        amex_metric(y_true.get_label(), y_pred),
        True
    )


def xy_set(x_data: pd.DataFrame, y_data: pd.DataFrame) -> tuple[pd.DataFrame]:
    x_t, x_v, y_t, y_v = model_selection.train_test_split(
        x_data, y_data,
        test_size=0.2,
        random_state=0
    )
    t_set = lgb.Dataset(x_t, y_t)
    v_set = lgb.Dataset(x_v, y_v, reference=t_set)

    return t_set, v_set, x_v, y_v


def show_result(model, x_value, y_value) -> pd.DataFrame:
    result = model.predict(x_value)
    score = amex_metric(y_value, result)
    print(model.params, score, sep="\n")
    fig = px.histogram(result, nbins=100)
    fig.show()
    importance = pd.DataFrame(
        data={
            "col": x_value.columns,
            "importance": model.feature_importance()
        }
    ).sort_values(["importance"])
    fig = px.bar(importance, x="col", y="importance")
    fig.show()

    return importance


def save_predict(
    model,
    data: pd.DataFrame,
    original: Path,
    save_path: Path,
    func=lambda x: x
) -> None:
    res_df = func(
        pd.DataFrame(
            data={"prediction": model.predict(data)},
            index=data.index
        )
    )
    sub_df = pd.read_csv(original, usecols=["customer_ID"])
    sub_df = sub_df.merge(res_df, left_on="customer_ID", right_index=True)
    print(sub_df.head(10))
    sub_df.to_csv(save_path, index=False)
