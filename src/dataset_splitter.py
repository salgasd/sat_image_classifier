import numpy as np
import pandas as pd
from numpy.typing import NDArray
from skmultilearn.model_selection.iterative_stratification import (
    IterativeStratification,
)


def stratify_shuffle_split_subsets(
    annotation: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_columns = ["image_name", "tags"]
    y_columns = list(annotation.select_dtypes("int").columns)

    all_x = annotation[x_columns].to_numpy()
    all_y = annotation[y_columns].to_numpy()

    train_indexes, else_indexes = _split(all_x, all_y, distribution=[1 - train_fraction, train_fraction])
    x_train, x_else = all_x[train_indexes], all_x[else_indexes]
    y_train, y_else = all_y[train_indexes], all_y[else_indexes]

    test_indexes, valid_indexes = _split(x_else, y_else, distribution=[0.5, 0.5])
    x_test, x_valid = x_else[test_indexes], x_else[valid_indexes]
    y_test, y_valid = y_else[test_indexes], y_else[valid_indexes]

    train_subset = pd.DataFrame(data=np.concatenate([x_train, y_train], axis=1), columns=x_columns + y_columns)
    valid_subset = pd.DataFrame(data=np.concatenate([x_valid, y_valid], axis=1), columns=x_columns + y_columns)
    test_subset = pd.DataFrame(data=np.concatenate([x_test, y_test], axis=1), columns=x_columns + y_columns)

    return train_subset, valid_subset, test_subset


def _split(
    xs: NDArray[np.float32],
    ys: NDArray[np.float32],
    distribution: list[float] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    stratifier = IterativeStratification(
        n_splits=2,
        sample_distribution_per_fold=distribution,
    )
    first_indexes, second_indexes = next(stratifier.split(X=xs, y=ys))

    return first_indexes, second_indexes
