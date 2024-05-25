from itertools import chain
import pandas as pd
import optuna
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from lightgbm import LGBMClassifier

class ObjectiveCV(object):
    def __init__(self, X_data, y_data, multilabel_data):
        self.X_data = X_data
        self.y_data = y_data
        self.multilabel_data = multilabel_data

    def __call__(self, trial):
        params = {
            "class_weight": "balanced",
            "objective": "binary",
            "verbosity": -1,
            "n_jobs": 2,
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 9),
            "n_estimators": trial.suggest_int("n_estimators", 50, 2500),  # n_trees
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.05, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.05, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 5000),
            "max_depth": trial.suggest_int("max_depth", 3, 13),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3),
        }

        params["num_leaves"] = 2 ** params["max_depth"]

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        result = []
        for train_index, test_index in cv.split(self.X_data, self.y_data):
            X_train, X_test = (
                self.X_data.iloc[train_index],
                self.X_data.iloc[test_index],
            )
            y_train, y_test = (
                self.y_data.iloc[train_index],
                self.y_data.iloc[test_index],
            )

            model = LGBMClassifier(**params, verbose=-1)
            model.fit(X_train.astype(float), y_train)
            y_pred = model.predict(X_test)
            f1_val = f1_score(y_test, y_pred, average="micro")

            result.append(f1_val)

        mean_f1 = np.mean(result)

        return mean_f1
    
class LGBMBinaryHierarchicalClassifier:
    def __init__(self, hierarchy: dict, trials: int, model=LGBMClassifier):
        """
        Initialize the Binary Hierarchical Classifier.

        Parameters:
        - hierarchy (dict): A dictionary representing the hierarchical structure of classes.
        - trials (int): Number of trials for hyperparameter optimization.
        - model (object): The classifier model to use. Default is LGBMClassifier.
        """
        self.hierarchy = hierarchy
        self.optuna_trials = trials
        self.classifiers = {}  # Dictionary to store classifiers for each node
        self.best_params = {}
        self.predictions = None  # Store final predictions
        self.lgbm_model = model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMBinaryHierarchicalClassifier":
        """
        Fit the classifier to the training data.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): Training data.
        - y (array-like, shape (n_samples,)): Target values.

        Returns:
        - self
        """
        self._fit_recursive(X, y, "Sample", self.hierarchy)

    def _modify_y(self, current_level_labels: list, y: np.ndarray) -> np.ndarray:
        """
        Transform labels based on the hierarchy.

        Parameters:
        - current_level_labels (2d list): List of lists of labels at the current level.
        - y (pandas.Series): Target values.

        Returns:
        - pandas.Series: Transformed target values.
        """
        orig_labels_sublist_1 = current_level_labels[0]
        orig_labels_sublist_2 = current_level_labels[1]

        new_labels_sublist_1, new_labels_sublist_2 = self._collapse_multilabels(
            orig_labels_sublist_1, orig_labels_sublist_2
        )

        new_labels_sublist_1 = new_labels_sublist_1 * len(orig_labels_sublist_1)
        new_labels_sublist_2 = new_labels_sublist_2 * len(orig_labels_sublist_2)

        labels_mapper_1 = dict(zip(orig_labels_sublist_1, new_labels_sublist_1))
        labels_mapper_2 = dict(zip(orig_labels_sublist_2, new_labels_sublist_2))

        labels_mapper = dict()
        for d in [labels_mapper_1, labels_mapper_2]:
            labels_mapper.update(d)

        combined_labels = list(chain.from_iterable(current_level_labels))
        y_transformed = y[y.isin(combined_labels)]
        y_transformed = y_transformed.map(labels_mapper)

        return y_transformed

    def _collapse_multilabels(
        self, labels_sublist_1: list, labels_sublist_2: list
    ) -> tuple:
        labels_sublist_2 = [labels_sublist_1[0] + 1]

        return labels_sublist_1, labels_sublist_2

    def _fit_recursive(
        self, X: np.ndarray, y: np.ndarray, node: str, tree: dict
    ) -> None:
        """
        Recursively fit the classifier to the data.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): Training data.
        - y (array-like, shape (n_samples,)): Target values.
        - node (str): Current node in the hierarchy.
        - tree (dict): Current subtree in the hierarchy.
        """
        tree_subset = tree[node]
        tree_subset_children = tree_subset["child_nodes"]

        # Collect labels from current level
        current_level_labels = []
        if tree_subset_children:
            for child in tree_subset_children.values():
                child_labels = child.get("labels")
                if child_labels:
                    current_level_labels.append(child_labels)

        # Train classifier if labels exist for the node
        if len(current_level_labels) == 0:
            return

        y_subset = self._modify_y(current_level_labels, y)
        X_subset = X.loc[y_subset.index]

        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        objective_cv = ObjectiveCV(X_subset, y_subset, y_subset)
        study_cv = optuna.create_study(direction="maximize")
        study_cv.optimize(objective_cv, n_trials=self.optuna_trials)
        best_params_cv = study_cv.best_params

        classifier = self.lgbm_model(**best_params_cv, verbose=-1)
        classifier.fit(X_subset, y_subset)
        children_nodes = tuple(tree_subset_children.keys())
        self.classifiers[children_nodes] = classifier
        self.best_params[children_nodes] = best_params_cv

        # Recursively fit child nodes
        if tree_subset_children:
            for child_node in tree_subset_children.keys():
                self._fit_recursive(X, y, child_node, tree_subset_children)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): Samples.

        Returns:
        - array-like, shape (n_samples,): Predicted class labels.
        """

        self.predictions = np.zeros((X.shape[0], 1))
        self._predict_recursive(X, "Sample", self.hierarchy, self.predictions)

        predictions_normalized = self._normalize_predictions()
        print(predictions_normalized.shape)
        idx_highest_proba = np.argmax(predictions_normalized, axis=1)
        res = idx_highest_proba + 1

        return res

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): Samples.

        Returns:
        - array-like, shape (n_samples,): Predicted class labels.
        """
        self.predictions = np.zeros((X.shape[0], 1))
        self._predict_recursive(X, "Sample", self.hierarchy, self.predictions)

        predictions_normalized = self._normalize_predictions()

        return predictions_normalized

    def _predict_recursive(
        self, X: np.ndarray, node: str, tree: dict, probs: np.ndarray
    ) -> None:
        """
        Recursively predict class probabilities for samples in X.

        Parameters:
        - X (array-like, shape (n_samples, n_features)): Samples.
        - node (str): Current node in the hierarchy.
        - tree (dict): Current subtree in the hierarchy.
        - probs (array-like, shape (n_samples,)): Predicted class probabilities.
        """
        tree_subset = tree[node]
        tree_subset_children = tree_subset["child_nodes"]
        if tree_subset_children is None:
            return

        children_nodes = tuple(tree_subset_children.keys())
        classifier = self.classifiers[children_nodes]

        clf_probabilities = classifier.predict_proba(X).reshape(-1, 2)
        self.predictions = np.concatenate((self.predictions, clf_probabilities), axis=1)

        # Recursively predict probabilities for child nodes
        if tree_subset_children:
            for child_node in tree_subset_children.keys():
                self._predict_recursive(
                    X, child_node, tree_subset_children, self.predictions
                )

    def _normalize_predictions(self) -> np.ndarray:
        """
        Normalize the predicted probabilities.

        Returns:
        - array-like, shape (n_samples,): Normalized predicted probabilities.
        """
        concat_predictions = self.predictions
        concat_predictions_normalized = concat_predictions.copy()

        for i in range(1, concat_predictions.shape[1] - 1, 2):
            concat_predictions_normalized[:, i + 1 :] *= concat_predictions[:, i][
                :, np.newaxis
            ]
        concat_predictions_normalized = np.hstack(
            (
                concat_predictions_normalized[:, :-1:2],
                concat_predictions_normalized[:, -1][:, np.newaxis],
            )
        )
        return concat_predictions_normalized