# %%
# standard lib
from pathlib import Path
# third party
import dask.dataframe as dd
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
# %%
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
fig.update_traces(opacity=0.80)
fig.show()
# %%


def multiplot(chart_title: str, data: pd.DataFrame, x_var: pd.Series):
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
        fig.update_layout(
            title=chart_title,
            xaxis_title="Date",
            yaxis_title="Normalized var",
            margin_l=10,
            margin_b=10,
            margin_t=30,
            height=450,
        )
    fig.show()
    print(f"{obj_list=}", f"{stb_list=}", sep="\n")


def create_categorichart(data: pd.DataFrame, target: int):
    for cols in (
        [col for col in data.columns if col[0] == "D"],
        [col for col in data.columns if col[0] == "S"],
        [col for col in data.columns if col[0] == "P"],
        [col for col in data.columns if col[0] == "B"],
        [col for col in data.columns if col[0] == "R"]
    ):
        multiplot(
            f"target={target}, line={len(cols)}",
            data[cols],
            data["S_2"]
        )


cid = train_unique.query("target==0")["customer_ID"].values[0]
train_personal = train_data.query(f"customer_ID=='{cid}'")
create_categorichart(train_personal, 0)
cid = train_unique.query("target==1")["customer_ID"].values[0]
train_personal = train_data.query(f"customer_ID=='{cid}'")
create_categorichart(train_personal, 1)
# %%
