import abc
from typing import List, Tuple

from Levenshtein import distance
import numpy as np
from scipy import sparse
from sklearn import exceptions as NotFittedError
from sklearn.feature_extraction.text import CountVectorizer


def create_sparce_from_diagonal(diag: np.ndarray) -> sparse.csr_matrix:
    """Creates sparse matrix with provided elements on the diagonal."""
    n = len(diag)
    return sparse.csr_matrix((diag, (list(range(n)), list(range(n)))), shape=(n, n))


def normalize_sparse_matrix(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Normalizes each row to one.

    Norms of rows aren't exactly because before division 0.1 is added
    just to make sure that 0 doesn't appear in the nominator.
    """
    sums_for_strings = matrix.sum(axis=1).A.flatten()
    normalization_matrix = create_sparce_from_diagonal(1 / (sums_for_strings + 0.1))
    return normalization_matrix.dot(matrix)


def compute_norms(matrix: sparse.csr_matrix) -> np.ndarray:
    """Computes norms for each row."""
    return np.sqrt(matrix.multiply(matrix).sum(axis=1).A).flatten()


def get_row_col_indices_from_flat_index(index: int, n_cols: int) -> Tuple[int, int]:
    """Translates index from flatten 2-dimensional array into row and column indices.
        Args:
            index: index of the flatten array
            n_cols: number of columns of the original array
    """
    row = index // n_cols
    col = index - row * n_cols
    return row, col


class NGramer(abc.ABC):
    def get_n_grams(self, text) -> List[str]:
        raise NotImplementedError


class GridNGramer(NGramer):
    def __init__(self, ngram_range: int):
        self._ngram_range = ngram_range

    def get_n_grams(self, text: str) -> List[str]:
        split_text = text.split()
        n = len(split_text)

        n_grams = []
        for n_gram_length in range(1, self._ngram_range + 1):
            for start_index in range(n - n_gram_length + 1):
                end_index = start_index + n_gram_length
                n_grams.append(' '.join(split_text[start_index:end_index]))

        return n_grams


class TextNormalizer(abc.ABC):
    def normalize(self, text: str) -> str:
        raise NotImplementedError


class LowerCase(TextNormalizer):
    def normalize(self, text: str) -> str:
        return text.lower()


class FuzzyMatcher:
    """
    This class splits the process of finding fuzzy matches into 2 phases:
    - approximate part
    - fuzzy part

    Thanks to that, it can work with large data.

    In approximate part, some number of candidates are selected from the corpus based on the cosine distance
    between a target string from the corpus and the searching string. In this part order of letters is ignored.

    In fuzzy part, the set of candidates found in the previous step are assessed by computing levenshtein distance.
    """
    def __init__(
        self,
        n_approximate: int,
        n_gramer: NGramer,
        vectorizer: CountVectorizer,
        normalizer: TextNormalizer,
    ):
        self._n_approximate = n_approximate
        self._n_gramer = n_gramer
        self._vectorizer = vectorizer
        self._normalizer = normalizer

        self._vocabulary = None
        self._X_strings = None
        self._norms = None
        self._fitted = False

    def fit(self, vocabulary: List[str]):
        """
            Args:
                vocabulary: the set of possible matching (target) strings
        """
        self._vocabulary = vocabulary

        # of shape (vocabulary size, dimension)
        self._X_strings = self._vectorizer.fit_transform(self._vocabulary)
        # of shape (vocabulary size, dimension)
        self._X_strings = normalize_sparse_matrix(self._X_strings)
        # of shape (vocabulary size,)
        self._norms = compute_norms(self._X_strings)

        self._fitted = True

    def get_matches(self, text: str, n_top: int, tokenize: bool) -> List[Tuple[int, str, str]]:
        """
            Args:
                text:
                n_top: number of best matches to return
                tokenize: whether tokenize given text
            Returns:
                [
                    (levenshtein distance between target string,
                    target string from corpus,
                    matching string from text),
                    (levenshtein distance between target string,
                    target string from corpus,
                    matching string from text),
                     ..
                ]
        """

        if not self._fitted:
            raise NotFittedError

        n_grams = [text]
        if tokenize:
            n_grams += self._n_gramer.get_n_grams(text)

        scored_matches = []

        for target_string, matched_string in self._get_approximate_matches(n_grams):
            d = distance(
                self._normalizer.normalize(target_string),
                self._normalizer.normalize(matched_string),
            )
            scored_matches.append((
                d,
                target_string,
                matched_string
            ))

        scored_matches = sorted(scored_matches, key=lambda x: x[0])
        return scored_matches[:n_top]

    def _get_approximate_matches(self, text_batch: List[str]) -> List[Tuple[str, str]]:
        batch_size = len(text_batch)

        # of shape (batch size, dimension)
        x_text = self._vectorizer.transform(text_batch)
        x_text = normalize_sparse_matrix(x_text)

        # of shape (batch size,)
        batch_norms = compute_norms(x_text)

        # of shape (vocabulary size, n-grams)
        similarities = self._X_strings.dot(x_text.T).A

        similarities = similarities / self._norms.reshape((-1, 1))
        similarities = similarities / batch_norms.reshape((1, -1))
        similarities = similarities.flatten()

        top_indices = np.argsort(similarities)[-self._n_approximate:]

        approximate_pairs = []

        for index in top_indices:
            string_index, col_index = get_row_col_indices_from_flat_index(index, batch_size)
            approximate_pairs.append((self._vocabulary[string_index], text_batch[col_index]))

        return approximate_pairs

    @classmethod
    def build_default(cls):
        return cls(
            n_approximate=50,
            n_gramer=GridNGramer(3),
            vectorizer=CountVectorizer(
                ngram_range=(1, 2),
                lowercase=True,
                binary=False,
                analyzer='char'
            ),
            normalizer=LowerCase()
        )
