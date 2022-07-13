# %%
# standard lib
import gc
from pathlib import Path
# third party
import dask.dataframe as dd
import pandas as pd
import numpy as np
from plotly import express as px  # kaleido==0.1.0.post1 (win11)
# from plotly import graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# %%
save_path = Path(r"..\Document\src")
train_data_path = Path(r"Data\feather_data\train_data.ftr")
train_labels_path = Path(r"Data\amex-default-prediction\train_labels.csv")
# %%
raw_data = dd.read_csv(Path(r"Data\amex-default-prediction\train_data.csv"))
print(raw_data.info(), raw_data.size, sep="\n")
raw_data.head()
del raw_data
gc.collect()
# %%
train_data = pd.read_feather(train_data_path)
train_labels = pd.read_csv(train_labels_path)
# %%
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
# fig.write_image(save_path / "target_histogram.svg", scale=10)
fig.show()
# %%


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
        margin_b=15,
        width=700,
        height=600,
    )
    fig.show()
    fig.write_image(
        save_path / f"correlation_matrix_{category}.svg", scale=10
    )


def principal_component_analysis(data: pd.DataFrame, category: str) -> None:
    ss = StandardScaler()
    pca = PCA()

    correlation_heatmap(data, category)
    st_data = ss.fit_transform(data.fillna(0))
    pca.fit(st_data)
    prop_var = np.cumsum(pca.explained_variance_ratio_)

    fig = px.scatter(prop_var)
    fig.update_layout(
        # template="plotly_dark",
        title={
            "text": f"proportion of variance {category}",
            "font": {
                "size": 22,
                "color": "black"
            },
            "x": 0.5,
            "y": 0.95,
        },
        margin_b=15,
    )
    fig.show()
    fig.write_image(
        save_path / f"prop_var_{category}.svg", scale=10
    )


train_data = (
    train_data
    .drop(["S_2", "D_63", "D_64"], axis=1)
    .set_index("customer_ID")
)
col_items = train_data.columns
for cat in ("R", "D", "S", "P", "B"):
    data = train_data[[col for col in col_items if col[0] == cat]]
    principal_component_analysis(data, cat)

    if cat == "R":
        break

# %%
