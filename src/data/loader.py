# Std Libs
import os

# Third Party Libs
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

DIR_DATASETS = Path(__file__).resolve().parent / 'datasets'
assert os.path.exists(DIR_DATASETS), (
    f'Datasets directory missing: {DIR_DATASETS}'
)

FP_ENB_DATA = DIR_DATASETS / "ENB2012_data.csv"


def load_cercles_ds(
) -> pd.DataFrame:
    """
    Returns a dataframe with 
    x1 (feature 1), x2 (feature 1), y (class)
    """
    # Cercles dataset
    fn = "cercles3.txt"
    fp_ds_c = DIR_DATASETS / fn
    assert os.path.exists(fp_ds_c)
    df_c = pd.read_csv(fp_ds_c, sep=" ")
    print(f"Read: {fn}")
    assert len(df_c) > 0, f'No row in file: {fp_ds_c}'

    # Create labels
    df_c["y"] = np.sqrt((df_c["x1"] ** 2) + df_c["x2"] ** 2)
    c1 = df_c["y"] < 1.5
    c2 = (df_c["y"] > 1.5) & (df_c["y"] <= 3.2)
    c3 = df_c["y"] > 3.2
    df_c["y"] = np.where(c1, 1, np.where(c2, 2, np.where(c3, 3, -1)))

    return df_c


def load_noeud_ds(
) -> pd.DataFrame:
    """
    Returns a dataframe with 
    x1 (feature 1), x2 (feature 1)
    """
    # Cercles dataset
    fn = "noeud.txt"
    fp_ds_n = DIR_DATASETS / fn
    assert os.path.exists(fp_ds_n)
    df = pd.read_csv(fp_ds_n, sep=" ")
    print(f"Read: {fn}")
    assert len(df) > 0, f'No row in file: {fp_ds_n}'

    df = df[["x1", "x2"]]
    
    return df


def get_enb_data(
    subset: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Read data
    __df = pd.read_csv(FP_ENB_DATA, sep=",")
    print(f"Read: {FP_ENB_DATA}")

    # feautres(mode)
    dict_list_features = {
        1: [1, 3, 6, 7, 8],  # no correlation
        2: [1, 3, 4, 6, 7, 8],  # some correlation but X still full rank
        3: [1, 2, 3, 4, 5, 6, 7, 8],  # correlation and X not full rank
    }
    assert subset in dict_list_features

    # Features
    cols_X_raw = [f'X{i}' for i in range(1, 9)]
    cols_y = [f'Y{i}' for i in range(1, 3)]
    assert set(__df.columns) == set(cols_X_raw) | set(cols_y)
    X_raw = __df[cols_X_raw].values.astype('float32') # Features
    print(f'Features: {X_raw.shape}')

    # Features Selection
    list_f = dict_list_features[subset]
    cols_X = [f'X{i}' for i in list_f]
    df_X = __df[cols_X]

    # Targets
    df_y = __df[cols_y] # Targets
    print(f'Targets: {df_y.shape}')

    return df_X, df_y


if __name__ == '__main__':
    # fp = Path('.') / 'data' / 'datasets'
    print(load_cercles_ds())
