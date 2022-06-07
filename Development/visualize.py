# %%
# standard lib
from pathlib import Path
# third party
import dask.dataframe as dd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

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
plt.figure(figsize=(12, 7))
for item, color in zip(train_labels["target"].unique(), ("blue", "red")):
    data = train_unique.query(f"target=={item}")
    sns.histplot(
        data["size"],
        bins=12,
        kde=False,
        alpha=0.5,
        color=color,
        label=f"target={item}"
    )
plt.show()
# %%
ratio = 20
obj_list = []
stb_list = []
cid = train_unique["customer_ID"][0]
train_personal = train_data.query(f"customer_ID=='{cid}'")
plt.figure(figsize=(12, 7))
for item in train_personal.columns[2:]:
    if train_personal[item].dtypes == "object":
        obj_list.append(item)
        continue
    if all(
        [(ave := train_personal[item].mean()) * (1 - ratio / 100) <=
         i < ave * (1 + ratio / 100)
         for i in train_personal[item]]
    ):
        stb_list.append(item)
        continue
    sns.lineplot(data=train_personal, x="S_2", y=item)
plt.show()
print(obj_list, stb_list, sep="\n")
print(train_data[obj_list])
print(train_data[stb_list])
# %%
