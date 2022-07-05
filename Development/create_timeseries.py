# Standard lib
import gc
from pathlib import Path
from dataclasses import dataclass, field
# Third party
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class Config():
    train_data_path: Path = Path(r"Data\feather_data\train_data.ftr")
    test_data_path: Path = Path(r"Data\feather_data\test_data.ftr")
    train_label_path: Path = Path(
        r"Data\amex-default-prediction\train_labels.csv"
    )
    category: tuple = ("D", "S", "P", "B", "R")
    remove_param: list = field(default_factory=lambda: ["S_2", "D_63", "D_64"])


CFG = Config()

train_data = (
    pd.read_feather(CFG.train_data_path)
    .set_index("customer_ID", drop=True)
    .drop(CFG.remove_param, axis=1)
)
train_label = pd.read_csv(CFG.train_label_path, index_col="customer_ID")


def time_series(data: pd.DataFrame, save_path: Path):
    df = pd.DataFrame()
    category_dict = {
        cat: [it for it in data.columns if it[0] == cat]
        for cat in CFG.category
    }

    for c_id, statements in tqdm(data.groupby("customer_ID")):
        df_cid = pd.DataFrame(data={"customer_ID": [c_id]})
        for ini, cat in category_dict.items():
            cat_ave = statements[cat].apply(
                lambda x: np.nanmean(x)
                if not all(x.isnull()) else np.nan
            )
            cat_bar = np.nanmean(cat_ave)
            cat_std = cat_ave.map(lambda x: np.square((x - cat_bar)))
            feat_cat = statements[cat_std.idxmax()]
            num = len(feat_cat)
            for i, (_, val) in enumerate(feat_cat.iteritems()):
                df_cid[f"{ini}_feat{13-num+i}"] = val
        df = pd.concat([df, df_cid])

        del df_cid
        gc.collect()

    df.reset_index(drop=True, inplace=True)
    df.to_csv(save_path)


time_series(train_data, CFG.train_data_path.parent / "train_time_series.ftr")
