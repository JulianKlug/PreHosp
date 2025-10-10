"""Custom preprocessing transformers for the analgesia predictor pipeline."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class MultiLabelTopKEncoder(BaseEstimator, TransformerMixin):
    """Encode multi-label categorical columns into binary indicator features.

    The transformer expects a pandas DataFrame with one or more string columns where each
    cell contains a delimited list of tokens (e.g., "A; B; C"). During ``fit`` the top ``k``
    most frequent tokens per column are retained. During ``transform`` the encoder emits a
    sparse binary matrix with one column per retained token indicating its presence on each
    row.
    """

    def __init__(
        self,
        top_k: int = 20,
        delimiters: Optional[Iterable[str]] = None,
        lowercase: bool = True,
    ) -> None:
        self.top_k = top_k
        self.delimiters = tuple(delimiters) if delimiters is not None else (";", ",")
        self.lowercase = lowercase
        self.columns_: List[str] = []
        self.vocabulary_: dict[str, List[str]] = {}
        self._token_to_index_: dict[str, dict[str, int]] = {}

    def fit(self, X, y=None):  # type: ignore[override]
        import pandas as pd

        if not hasattr(X, "columns"):
            raise TypeError("MultiLabelTopKEncoder expects a pandas DataFrame input")

        df = pd.DataFrame(X)
        self.columns_ = list(df.columns)
        self.vocabulary_.clear()
        self._token_to_index_.clear()

        for column in self.columns_:
            counter: Counter[str] = Counter()
            series = df[column].dropna().astype(str)
            for cell in series:
                tokens = self._tokenize(cell)
                if not tokens:
                    continue
                counter.update(tokens)
            top_tokens = [token for token, _ in counter.most_common(self.top_k)]
            self.vocabulary_[column] = top_tokens
            self._token_to_index_[column] = {token: idx for idx, token in enumerate(top_tokens)}
        return self

    def transform(self, X):  # type: ignore[override]
        import pandas as pd

        if not self.columns_:
            raise RuntimeError("The encoder has to be fitted before calling transform().")

        df = pd.DataFrame(X, columns=self.columns_)
        n_samples = df.shape[0]
        matrices = []

        for column in self.columns_:
            tokens_per_row = df[column].fillna("").astype(str).map(self._tokenize)
            vocab = self.vocabulary_.get(column, [])
            if not vocab:
                continue
            token_to_index = self._token_to_index_[column]
            data: List[float] = []
            row_idx: List[int] = []
            col_idx: List[int] = []
            for i, token_set in enumerate(tokens_per_row):
                for token in token_set:
                    feature_idx = token_to_index.get(token)
                    if feature_idx is None:
                        continue
                    data.append(1.0)
                    row_idx.append(i)
                    col_idx.append(feature_idx)
            if data:
                matrix = sparse.coo_matrix(
                    (data, (row_idx, col_idx)),
                    shape=(n_samples, len(vocab)),
                    dtype=np.float32,
                ).tocsr()
            else:
                matrix = sparse.csr_matrix((n_samples, len(vocab)), dtype=np.float32)
            matrices.append(matrix)

        if matrices:
            return sparse.hstack(matrices, format="csr")
        return sparse.csr_matrix((n_samples, 0), dtype=np.float32)

    def get_feature_names_out(self, input_features=None):  # type: ignore[override]
        feature_names: List[str] = []
        for column in self.columns_:
            for token in self.vocabulary_.get(column, []):
                feature_names.append(f"{column}__{token}")
        return np.array(feature_names, dtype=object)

    def _tokenize(self, value: str) -> set[str]:
        raw = value.strip()
        if not raw or raw.lower() in {"nan", "<na>", "none"}:
            return set()
        tokens = [raw]
        for delimiter in self.delimiters:
            split_tokens: List[str] = []
            for token in tokens:
                split_tokens.extend(token.split(delimiter))
            tokens = split_tokens
        cleaned = []
        for token in tokens:
            stripped = token.strip()
            if not stripped:
                continue
            cleaned.append(stripped.lower() if self.lowercase else stripped)
        return set(cleaned)
