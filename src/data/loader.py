# Std Libs
import os

# Third Party Libs
from pathlib import Path

import numpy as np
import pandas as pd

DIR_DATASETS = Path(__file__).resolve().parent / 'datasets'
assert os.path.exists(DIR_DATASETS), (
    f'Datasets directory missing: {DIR_DATASETS}'
)

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


if __name__ == '__main__':
    # fp = Path('.') / 'data' / 'datasets'
    print(load_cercles_ds())