from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class AnalgesiaRecommender:
    """Batch reinforcement-learning agent for analgesia dosing."""

    def __init__(
        self,
        numeric_features: Iterable[str],
        categorical_features: Iterable[str],
        action_table: pd.DataFrame,
        random_state: int = 42,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.feature_columns = self.numeric_features + [
            col for col in self.categorical_features if col not in self.numeric_features
        ]
        self.action_table = action_table.drop_duplicates("action_idx").set_index("action_idx")
        self.random_state = random_state

        self.state_transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    self.numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32),
                            ),
                        ]
                    ),
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )

        self.action_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=400,
            l2_regularization=0.1,
            random_state=random_state,
        )

        self._action_feature_matrix: Optional[np.ndarray] = None
        self._num_actions: Optional[int] = None
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "AnalgesiaRecommender":
        states = train_df[self.feature_columns]
        actions = train_df[["action_idx"]]
        rewards = train_df["reward"].astype(int)

        state_features = self.state_transformer.fit_transform(states)
        self.action_encoder.fit(actions)
        action_features = self.action_encoder.transform(actions)

        X = np.hstack([state_features, action_features]).astype(np.float32)
        y = rewards.to_numpy(dtype=np.int32)

        self.model.fit(X, y)

        action_names = list(actions.columns)
        action_ids = pd.DataFrame(
            np.arange(self.action_encoder.categories_[0].shape[0]).reshape(-1, 1),
            columns=action_names,
        )
        self._action_feature_matrix = self.action_encoder.transform(action_ids).astype(np.float32)
        self._num_actions = self._action_feature_matrix.shape[0]
        self._is_fitted = True
        return self

    def predict_action(self, state_df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_fitted()
        state_features = self.state_transformer.transform(state_df[self.feature_columns]).astype(np.float32)
        num_samples = state_features.shape[0]
        repeated_states = np.repeat(state_features, self._num_actions, axis=0)
        tiled_actions = np.tile(self._action_feature_matrix, (num_samples, 1))
        q_inputs = np.hstack([repeated_states, tiled_actions])
        q_values = self.model.predict_proba(q_inputs)[:, 1].reshape(num_samples, self._num_actions)
        best_actions = np.argmax(q_values, axis=1)
        best_values = q_values[np.arange(num_samples), best_actions]
        result = pd.DataFrame(
            {
                "action_idx": best_actions,
                "expected_success": best_values,
            },
            index=state_df.index,
        )
        return result

    def recommend(self, state_df: pd.DataFrame) -> pd.DataFrame:
        predictions = self.predict_action(state_df)
        enriched = predictions.join(
            self.action_table[
                [
                    "fent_band",
                    "ket_band",
                    "fent_label",
                    "ket_label",
                    "recommended_fent_mg",
                    "recommended_ket_mg",
                ]
            ],
            on="action_idx",
            how="left",
        )
        return enriched

    def evaluate(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
        self._ensure_fitted()

        train_states = train_df[self.feature_columns]
        train_actions = train_df[["action_idx"]]
        train_features = self._combine_state_action(train_states, train_actions)
        train_rewards = train_df["reward"].to_numpy()
        train_probs = self.model.predict_proba(train_features)[:, 1]
        metrics: Dict[str, float] = {}
        if len(np.unique(train_rewards)) > 1:
            metrics["train_auc"] = float(roc_auc_score(train_rewards, train_probs))
        else:
            metrics["train_auc"] = float("nan")

        test_states = test_df[self.feature_columns]
        test_actions = test_df[["action_idx"]]
        observed_success = float(test_df["reward"].mean())
        metrics["observed_success_rate"] = observed_success

        expected = self.predict_action(test_states)
        metrics["expected_policy_value"] = float(expected["expected_success"].mean())

        matched_mask = expected["action_idx"].to_numpy() == test_df["action_idx"].to_numpy()
        metrics["matched_coverage"] = float(matched_mask.mean())
        matched_rewards = test_df.loc[matched_mask, "reward"].to_numpy()
        metrics["matched_cases"] = float(matched_mask.sum())
        metrics["matched_success_rate"] = (
            float(matched_rewards.mean()) if matched_rewards.size else float("nan")
        )

        behaviour_stats = train_df.groupby("action_idx")["reward"].agg(["mean", "count"])  # type: ignore[attr-defined]
        best_action_idx = int(behaviour_stats["mean"].idxmax())
        metrics["best_historical_action"] = float(best_action_idx)
        metrics["best_historical_success_rate"] = float(behaviour_stats.loc[best_action_idx, "mean"])
        metrics["best_historical_support"] = float(behaviour_stats.loc[best_action_idx, "count"])

        test_features = self._combine_state_action(test_states, test_actions)
        test_probs = self.model.predict_proba(test_features)[:, 1]
        if len(np.unique(test_df["reward"])) > 1:
            metrics["test_auc_logged_actions"] = float(roc_auc_score(test_df["reward"], test_probs))
        else:
            metrics["test_auc_logged_actions"] = float("nan")

        return metrics

    def _combine_state_action(self, states: pd.DataFrame, actions: pd.DataFrame) -> np.ndarray:
        state_features = self.state_transformer.transform(states).astype(np.float32)
        action_features = self.action_encoder.transform(actions).astype(np.float32)
        return np.hstack([state_features, action_features])

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("The recommender has not been fitted yet.")
        if self._action_feature_matrix is None or self._num_actions is None:
            raise RuntimeError("Action encoder was not initialised correctly.")
