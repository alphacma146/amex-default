# standard lib
from pathlib import Path
import gc
# third party
from tqdm import tqdm
import pandas as pd
import numpy as np


def create_feather(input_path: Path, save_path: Path):
    '''
    csvからfeather形式に変換する
    '''
    df = pd.DataFrame()
    with pd.read_csv(input_path, chunksize=10000) as reader:
        for chunk in tqdm(reader):
            for column in chunk.columns:
                data_type = chunk[column].dtype
                if data_type == np.float64:
                    chunk[column] = chunk[column].astype(np.float16)
                elif data_type == np.int64:
                    chunk[column] = chunk[column].astype(np.int8)
            df = pd.concat([df, chunk])

            gc.collect()

        df.to_feather(save_path)


if __name__ == "__main__":
    path_list = [
        Path(r"Data\amex-default-prediction\train_data.csv"),
        Path(r"Data\amex-default-prediction\test_data.csv")
    ]
    feather_path = Path(r"Data\feather_data")
    for path in path_list:
        create_feather(path, (feather_path / path.name).with_suffix(".ftr"))
