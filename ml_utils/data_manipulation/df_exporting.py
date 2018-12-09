from typing import Any, Dict, List

import numpy as np
import pandas as pd


def df_to_dict_by_recursion(
    df: pd.DataFrame, 
    levels: List[List[int]], 
    level_names: List[List[str]],
    group_names: List[str],
    vector_name: str,
    n_level: int=0
) -> Dict[str, Any]:  
    """
    It extracts multindexed 'df' into dictionary whose 
    nested structure is determined by the passed arguments. 
    Columns of 'df' are returned in the lowest level as 
    'vector_name': list of values from all columns. 
    
    Let's assume that n is the number of levels of index.
    
    parameters
    ----------
    df: must satisfy n > 1
    
    levels: determines grouping of indices, is of the form 
    [[0, 1,..], [.., n-2, n-1]]
    
    level_names: must have the same structure as  'levels', basically
    it gives each level name of the key in the resulting dict
    
    group_names: must satisfy len(levels) == len(level_names)
    
    vector_name: key of the resulting vector
    
    n_level: integer between 0 and n - 1. It is accumulator used for 
    the purpose of recursion 
    
    examples
    --------
    Let us have the following 'df'
    
    index      v1 v2
    a1 b1 c1   1  2 
    a1 b1 c2   5  6
    a1 b3 c3   9  10
    a2 b1 c2   11 12
    
    df_to_dict_by_recursion(
        df=df,
        levels=[[0, 1], [2]],
        level_names=[['a', 'b'], ['c']],
        group_names=['g1', 'g2'],
        vector_name='values' 
    ) -> 
    {
        'g1': [
            {
                'a': 'a1',
                'b': 'b1',
                'g2': [
                    {
                        'c': 'c1',
                        'values': [[1, 2]]
                    },
                    {
                        'c': 'c2',
                        'values': [[5, 6]]
                    }
                ]
            },
            {
                'a': 'a1',
                'b': 'b3',
                'g2': [
                    {
                        'c': 'c3',
                        'values': [[9, 10]]
                    }
                ]
            },
            {
                'a': 'a2',
                'b': 'b1',
                'g2': [
                    {
                        'c': 'c2',
                        'values': [[11, 12]]
                    }
                ]
            }
        ]
    }
    """
    if n_level == 0:
        df = df.copy()
        df[vector_name] = df.apply(lambda x: x.values.tolist(), axis=1)
    
    current_levels = levels[n_level]
    current_level_names = level_names[n_level]
    data = []
    for labels, g_df in df.groupby(level=current_levels):
        if not isinstance(labels, tuple):
            labels = (labels, )
        d = {}
        for name, label in zip(current_level_names, labels):
            d[name] = label
        
        if len(levels) - 1 == n_level:
            d[vector_name] = g_df[vector_name].values.tolist()
        else:
            d[group_names[n_level + 1]] = df_to_dict_by_recursion(g_df, 
                                          levels, 
                                          level_names,
                                          group_names, 
                                          vector_name,
                                          n_level + 1)
        data.append(d)
    
    if n_level == 0:
        return {group_names[n_level]: data}
    
    return data


def df_to_dict(
    df: pd.DataFrame,
    level_priorities: List[int],
    level_names: List[str],
    group_names: List[str],
    vector_name: str
):
    """
    Its purpose is the same as in 'df_to_dict_by_recursion', but
    passed arguments makes it easier to use. 
    
    Let's assume that n is the number of levels of index.
    
    parameters
    ----------
    df: must satisfy n > 1
    
    level_priorities: must satisfy len(level_priorities) == n, 
    the bigger number a level has the in higher position it will end up
    in returned dictionary
    
    level_names: must satisfy len(level_names) == n
    in contract 'df_to_dict_by_recursion', it has the flat
    structure, ['level1', 'level2', .., 'leveln']
    
    group_names: must satisfy len(group_names) == len(set(level_priorities))
    
    vector_name: as in 'df_to_dict_by_recursion'
    
    examples
    --------
    Let us have the following 'df'
    
    index      values
    a1 b1 c1   1  2 
    a1 b1 c2   5  6
    a1 b3 c3   9  10
    a2 b1 c2   11 12
    
    df_to_dict_by_recursion(
        df=df,
        levels=[[0, 1], [2]],
        level_names=[['a', 'b'], ['c']],
        group_names=['g1', 'g2'],
        vector_name='values' 
    )
    
    is equal to 
    
    df_to_dict(
        df=df, 
        level_priorities=[1, 1, 0], # [99, 99, -1] would also work
        level_names=['a', 'b', 'c'],
        group_names=['g1', 'g2'],
        vector_name='values'
    )
    """
    df = df.copy()
    priority_indices = np.argsort(level_priorities)[::-1].tolist()
    df.index = df.index.reorder_levels(priority_indices)
    
    levels_ = []
    level_names_ = []
    last_value = None
    for i, index in enumerate(priority_indices):
        if last_value == level_priorities[index]:
            levels_[-1].append(i)
            level_names_[-1].append(level_names[index])
        else:
            levels_.append([i])
            level_names_.append([level_names[index]])
        last_value = level_priorities[index]
    return df_to_dict_by_recursion(
        df, 
        levels_, 
        level_names_, 
        group_names,
        vector_name
    )
