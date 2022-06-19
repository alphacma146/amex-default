# %%
# standard lib
from pathlib import Path
# third party
import dask.dataframe as dd
import pandas as pd
import numpy as np
from plotly import express as px  # kaleido==0.1.0.post1 (win11)
from plotly import graph_objects as go
# %%
save_path = Path(r"..\Document\src")
train_data_path = Path(r"Data\feather_data\train_data.ftr")
train_labels_path = Path(r"Data\amex-default-prediction\train_labels.csv")
# %%
raw_data = dd.read_csv(Path(r"Data\amex-default-prediction\train_data.csv"))
print(raw_data.info(), raw_data.size, sep="\n")
raw_data.head()
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


def multiplot(target: int, label: str, data: pd.DataFrame, x_var: pd.Series):
    ratio = 1
    obj_list = []
    stb_list = []
    fig = go.Figure()
    for item in data.columns[1:]:
        if item == "S_2":
            continue
        if data[item].dtypes == "object":
            obj_list.append(item)
            continue
        if all(
            [(ave := data[item].mean()) * (1 - ratio / 100) <=
             i < ave * (1 + ratio / 100)
             for i in data[item]]
        ):
            stb_list.append(item)
            continue
        fig.add_trace(
            go.Scatter(x=x_var, y=data[item], name=item)
        )
    feat_num = len(data.columns[1:]) - len(obj_list) - len(stb_list)
    fig.update_layout(
        title=f"{label} target={target}, line={feat_num}",
        xaxis_title="Date",
        yaxis_title="Normalized var",
        margin_l=10,
        margin_b=10,
        margin_t=30,
        height=450,
    )
    fig.show()
    # fig.write_image(save_path / f"transition_{label}_{target}.svg")
    print(f"{obj_list=}", f"{stb_list=}", sep="\n")


for target in train_labels["target"].unique():
    id_df = train_unique.query(f"target=={target}")["customer_ID"]
    cid = id_df.values[np.random.randint(0, len(id_df) + 1)]
    vis_data = train_data.query(f"customer_ID=='{cid}'")
    for feature, cols in (
        ("Delinquency", [col for col in train_data.columns if col[0] == "D"]),
        ("Spend", [col for col in train_data.columns if col[0] == "S"]),
        ("Payment", [col for col in train_data.columns if col[0] == "P"]),
        ("Balance", [col for col in train_data.columns if col[0] == "B"]),
        ("Risk", [col for col in train_data.columns if col[0] == "R"])
    ):
        multiplot(
            target, feature,
            vis_data[cols],
            vis_data["S_2"]
        )

# %%
