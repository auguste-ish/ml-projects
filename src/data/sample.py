from typing import List
import numpy as np
import pandas as pd


def generate_gaussian_mixture_sample(
    means: np.ndarray,
    covs: List[np.ndarray],
    size: int,
) -> pd.DataFrame:
    """
    Generate 2 dimension gaussian mixture sample
    Returns a dataframe with the following columns:
    - x1 (feature 1)
    - x2 (feature 2)
    - y: class corresponding to the gaussian distribution the point was sampled from 
    """
    # Classes, Features
    k, d = means.shape
    assert d == 2

    assert means.shape[0] == len(covs)
    assert all([c.shape == (d, d) for c in covs])

    # Weights of mixtures
    weights = np.random.randint(low=1, high=11, size=k)
    weights = weights / weights.sum()

    # Generate Sample
    shape_sample = (size, d)
    dict_samples_raw = {
        i: np.random.multivariate_normal(
            mean=means[i],
            cov=covs[i],
            size=size,
            check_valid="raise",
        )
        for i in range(k)
    }

    classes = np.random.choice(a=k, size=size, replace=True)
    sample = np.zeros(shape_sample)
    for i in range(k):
        gp_indexes = classes == i
        sample[gp_indexes] = dict_samples_raw[i][gp_indexes]

    # df
    df_gm = pd.DataFrame(sample, columns=["x1", "x2"])
    df_gm["y"] = classes
    return df_gm