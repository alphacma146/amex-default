# %%
# standard lib
# import gc
from pathlib import Path
# third party
# import dask.dataframe as dd
import pandas as pd
import numpy as np
from plotly import express as px  # kaleido==0.1.0.post1 (win11)
# from plotly import graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


CREATE_PICTURE = True

# %%
save_path = Path(r"..\Document\src")
train_data_path = Path(r"Data\feather_data\train_data.ftr")
test_data_path = Path(r"Data\feather_data\test_data.ftr")
train_labels_path = Path(r"Data\amex-default-prediction\train_labels.csv")
# raw_data = dd.read_csv(Path(r"Data\amex-default-prediction\train_data.csv"))
# print(raw_data.info(), raw_data.size, sep="\n")
# raw_data.head()
# del raw_data
# gc.collect()
train_data = pd.read_feather(train_data_path)
train_labels = pd.read_csv(train_labels_path)
# %%
# histogram statements
train_unique = train_data.groupby(
    "customer_ID",
    as_index=False
).size()
train_unique = pd.merge(
    train_unique,
    train_labels,
    left_on="customer_ID",
    right_on="customer_ID"
)
fig = px.histogram(
    train_unique,
    x="size",
    color="target"
)
fig.update_layout(barmode='overlay')
# fig.update_yaxes(range=(0, 10000))
fig.update_traces(opacity=0.80)
fig.update_layout(
    margin_l=10,
    margin_b=10,
    margin_t=30,
    height=450,
)
if CREATE_PICTURE:
    fig.write_image(save_path / "target_histogram.svg", scale=10)
fig.show()

del train_unique
# %%
# time scale


def vis_category_timeseries(data: pd.DataFrame, column: str):

    fig = px.line(
        data,
        line_group="customer_ID",
        x="S_2",
        y=column,
        color="target"
    )
    fig.update_traces(opacity=0.80)
    fig.update_layout(
        title={
            "text": col,
            "font": {
                "size": 22,
                "color": "black"
            },
            "x": 0.5,
            "y": 0.95,
        },
        margin_l=15,
        margin_b=15,
        margin_t=50,
        height=450,
        xaxis={"title": None},
        yaxis={"title": None}
    )
    if CREATE_PICTURE:
        fig.write_image(save_path / f"timescale_{column}.svg", scale=10)
    fig.show()


target_ids = pd.DataFrame(columns=["customer_ID", "target"])
for target in (0, 1):
    df = train_labels.query(f"target=={target}").sample(n=100)
    target_ids = pd.concat([target_ids, df])

train_timescale = train_data.merge(
    target_ids.set_index("customer_ID", drop=True),
    how="inner",
    on="customer_ID"
)

ignore_cols = ["customer_ID", "S_2", "target"]
for col in train_timescale.columns:
    if col in ignore_cols:
        continue

    vis_category_timeseries(
        train_timescale[ignore_cols + [col]], col
    )

del train_timescale
# %%
# 主成分分析


def correlation_heatmap(data: pd.DataFrame, category) -> None:
    correlation_matrix = data.corr("spearman")
    fig = px.imshow(correlation_matrix)
    fig.update_layout(
        # template="plotly_dark",
        title={
            "text": f"parameter category {category}",
            "font": {
                "size": 22,
                "color": "black"
            },
            "x": 0.5,
            "y": 0.95,
        },
        margin_l=5,
        margin_b=10,
        width=700,
        height=600,
    )
    if CREATE_PICTURE:
        fig.write_image(
            save_path / f"correlation_matrix_{category}.svg", scale=10
        )
    fig.show()


def principal_component_analysis(
        data: pd.DataFrame,
        category: str
) -> pd.DataFrame:
    ss = StandardScaler()
    pca = PCA(svd_solver="full")
    data = data.fillna(0)

    correlation_heatmap(data, category)
    st_data = ss.fit_transform(data)
    pca.fit(st_data)
    contribution_ratio = np.cumsum(pca.explained_variance_ratio_)

    fig = px.line(contribution_ratio, markers=True)
    fig.update_layout(
        # template="plotly_dark",
        title={
            "text": f"contribution ratio {category}",
            "font": {
                "size": 22,
                "color": "black"
            },
            "x": 0.5,
            "y": 0.95,
        },
        margin_b=15,
    )
    if CREATE_PICTURE:
        fig.write_image(
            save_path / f"contribution_ratio_{category}.svg", scale=10
        )
    fig.show()


train_pca = (
    train_data
    .drop(["S_2", "D_63", "D_64"], axis=1)
    .set_index("customer_ID")
)

col_items = train_pca.columns
ret_dict = {}
for cat in ("D", "S", "P", "B", "R"):
    data = train_pca[[col for col in col_items if col[0] == cat]]
    principal_component_analysis(data, cat)

    """ if cat == "R":
        break """

del train_pca
# %%
