from typing import Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline


"""
For sparse NN

keras=2.0.8
tensorflow=1.4.0
"""


class Preprocessor(BaseEstimator, TransformerMixin):
    _TEXT_COL = '...'
    _NUMERIC_COL1 = '...'
    _NUMERIC_COL2 = '...'
    
    def __init__(
        self,
        n_text_features: int,
        n_gram_range: Tuple[int, int],
        use_tf_idf: bool,
        use_numeric_col1: bool,
        use_numeric_col2: bool,
    ):    
        self._n_text_features = n_text_features
        self._n_gram_range = n_gram_range
        self._use_tf_idf = use_tf_idf
        self._use_numeric_col1 = use_numeric_col1
        self._use_numeric_col2 = use_numeric_col2
        
        if self._use_tf_idf_short:
            self._vectorizer = TfidfVectorizer(
                ngram_range=self._n_gram_range,
                max_features=self._n_text_features,
            )
        else:
            self._vectorizer = CountVectorizer(
                ngram_range=self._n_gram_range,
                max_features=self._n_text_features,
            )

        self._pipeline_list = []

        if use_numeric_col1:
            pipeline_for_numeric_col1 = self._from_column(
                [self._NUMERIC_COL1],
                FunctionTransformer(self._to_records, validate=False),
                DictVectorizer(),
            )
            self._pipeline_list.append(pipeline_for_numeric_col1)

        if use_numeric_col2:
            pipeline_for_numeric_col2 = self._from_column(
                [self._NUMERIC_COL1],
                FunctionTransformer(self._apply_some_numeric_operation, validate=False),
                FunctionTransformer(self._to_records, validate=False),
                DictVectorizer(),)
            self._pipeline_list.append(pipeline_for_numeric_col2)

        self._pipeline_list += [self._from_column(self._TEXT_COL, self._vectorizer)]
        
        self._preprocessor = make_union(*self._pipeline_list)
        
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        self._preprocessor.fit(df)
        return self
    
    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        return self._preprocessor.transform(df)
    
    def _from_column(self, col: Union[str, List[str]], *vec) -> Pipeline:
        return make_pipeline(FunctionTransformer(itemgetter(col), validate=False), *vec)
    
    def _apply_some_numeric_operation(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
    
    def _to_records(self, df: pd.DataFrame) -> List[Dict[str, float]]:
        return df.to_dict(orient='records')
