import gc
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SAVE_FLAG = True
COMP_RATE = 0.8
CATEGORY_LIST = ("D", "S", "P", "B", "R")

train_data_path = Path(r"Data\feather_data\train_data.ftr")
test_data_path = Path(r"Data\feather_data\test_data.ftr")


def principal_component_analysis(
        data: pd.DataFrame,
        category: str
) -> pd.DataFrame:
    ss = StandardScaler()
    pca = PCA(n_components=COMP_RATE, svd_solver="full")
    data = data.fillna(0)

    st_data = ss.fit_transform(data)
    pca.fit(st_data)
    contribution_ratio = np.cumsum(pca.explained_variance_ratio_)

    feature_data = pd.DataFrame(
        pca.transform(st_data),
        columns=[f"PC_{category}_{n}" for n in range(len(contribution_ratio))]
    )

    for column in feature_data.columns:
        if feature_data[column].dtype == np.float64:
            feature_data[column] = feature_data[column].astype(np.float16)

    # feature_data["customer_ID"] = data.index
    # feature_data.reset_index(drop=True, inplace=True)

    """ if cat == "R":
        break """

    return pca, feature_data


def save_as_feather(folder_path: Path, save_path: Path, index):
    df = pd.DataFrame({"customer_ID": index})
    for item in folder_path.glob("*"):

        df = df.merge(
            pd.read_feather(item),
            how="left",
            left_index=True,
            right_index=True,
            copy=False
        )

        gc.collect()

    df.to_feather(save_path)


if __name__ == "__main__":

    # 学習
    # train_data変換

    train_data = (
        pd.read_feather(train_data_path)
        .drop(["S_2", "D_63", "D_64"], axis=1)
        .set_index("customer_ID")
    )
    col_items = train_data.columns
    ret_dict = {}
    for cat in tqdm(CATEGORY_LIST):
        data = train_data[[col for col in col_items if col[0] == cat]]
        ret_dict[cat], pca_data = principal_component_analysis(data, cat)

        if SAVE_FLAG:
            pca_data.to_feather(
                train_data_path.parent /
                (r"train_pca\train_pca" + f"_{cat}.ftr")
            )
        else:
            next

    data_index = train_data.index
    del train_data
    gc.collect()

    save_as_feather(
        train_data_path.parent / r"train_pca",
        train_data_path.parent / "train_pca_data.ftr",
        data_index
    )

    # test_data変換

    test_data = (
        pd.read_feather(test_data_path)
        .drop(["S_2", "D_63", "D_64"], axis=1)
        .set_index("customer_ID")
    )
    col_items = test_data.columns

    for cat in tqdm(CATEGORY_LIST):
        if not SAVE_FLAG:
            break

        data = test_data[[col for col in col_items if col[0] == cat]]
        data = data.fillna(0)
        st_data = StandardScaler().fit_transform(data)
        pca = ret_dict[cat]
        feature_data = pd.DataFrame(
            pca.transform(st_data),
            columns=[
                f"PC_{cat}_{n}" for n in range(
                    len(pca.explained_variance_ratio_)
                )
            ]
        )

        for column in feature_data.columns:
            if feature_data[column].dtype == np.float64:
                feature_data[column] = feature_data[column].astype(np.float16)

        feature_data.to_feather(
            test_data_path.parent / (r"test_pca\test_pca" + f"_{cat}.ftr")
        )

    data_index = test_data.index
    del test_data
    gc.collect()

    save_as_feather(
        test_data_path.parent / r"test_pca",
        test_data_path.parent / "test_pca_data.ftr",
        data_index
    )
