# %%
# Standard libe
# Third party
import numpy as np
import pandas as pd
import scipy as sp
import plotly.express as px
# self made
from amex_base import Config

CFG = Config()
# %%
result_list = [p for p in CFG.result_folder_path.glob("*")]
sample_submission = pd.read_csv(
    CFG.sample_submission_path,
    usecols=["customer_ID"]
)
# %%
sample_submission["prediction"] = 0
for result in result_list:
    df = pd.read_csv(result)
    df["prediction"] = np.clip(df["prediction"], 0, 1)
    sample_submission["prediction"] += df["prediction"]

sample_submission["prediction"] /= len(result_list)
sample_submission["prediction"] **= 0.9
sample_submission.to_csv(
    CFG.result_folder_path.parent / "result_mean.csv",
    index=None
)
fig = px.histogram(sample_submission, x="prediction", nbins=100)
fig.show()
# %%
