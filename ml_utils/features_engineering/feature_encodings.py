from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def categorical_encoding_cv(
        df,
        cat_name,
        target_name,
        encoded_name=None,
        n_folds=5,
        fn=np.mean,
        seed=1,
        regression_problem=True
):
    """
    It performs encoding of categorical features specified
    by 'fn'. The latter can be 'mean' function or something more
    complicated.

    Calculation is done in cross-validation fashion.
    For example if n_folds=3 (F1, F2, F3), encodings are calculated
    on (F1, F2), (F1, F3), (F2, F3) and then they are averaged.

    parameters
    ----------
    df: pandas DataFrame
    cat_name: categorical column
    target_name: target column
    encoded_name: encoding column
    n_folds: number of splits for cross-validation
    fn: mapping iterable -> float
    seed: random state for splitting
    regression_problem: bool, if target variable is continuous

    returns
    -------
    df: original df extended by 'encoded_name' column
    encoding: dict, label -> encoded value
    """
    if regression_problem:
        cv = KFold(n_folds, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)

    global_encoding = fn(df[target_name])
    scores = pd.DataFrame(0,
                          columns=['fold' + str(i) for i in range(n_folds)],
                          index=df[cat_name].unique())

    col = pd.Series(np.nan, index=df.index)

    for i, (tr_inds, val_inds) in enumerate(cv.split(df, df[target_name])):
        encodings = df.iloc[tr_inds].groupby(cat_name)[target_name].apply(fn)
        scores.iloc[:, i] = scores.index.to_series().map(encodings)
        col.iloc[val_inds] = df[cat_name].iloc[val_inds].map(encodings).values

    df[encoded_name] = col.values
    df[encoded_name].fillna(global_encoding, inplace=True)

    return df, defaultdict(lambda: global_encoding, scores.fillna(global_encoding).mean(axis=1).to_dict())
