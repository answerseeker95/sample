import abc
import numpy as np


class ClassifierModel(abc.ABC):
    """Base model class"""
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class InformationCriterion:
    GINI = "gini"


class Node:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.left = left
        self.right = right
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.val = value


class CARTClassifier(ClassifierModel):
    """Follows https://en.wikipedia.org/wiki/Decision_tree_learning description of CART
    Learning algorithm follows:
    1. Initialize tree
    2. Iterate while information gain and maximal tree size not reached
      a. Try prediction
      b. Each split based on single feature, calculate information gain or Gini coefficient
      c. Recursively split nodes
    """
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.n_classes = None
        self.n_features = None
        self.class_val_for_id = None
        self.tree = None
        self.criterion = InformationCriterion.GINI

    def _validate_input(self, X, y):
        assert len(X.shape) == 2 and len(y.shape == 1)

    def fit(self, X, y):
        self._validate_input(X, y)
        self.n_classes = len(set(y))
        self.class_val_for_id = {i: c for i, c in enumerate(set(y))}
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict_node(xi) for xi in X]

    def _predict_node(self, features):
        node = self.tree
        while node.right or node.left:
            feature_val = features[node.feature_idx]
            if isinstance(node.threshold, str):
                # categorical feature: membership
                if feature_val == node.threshold:
                    node = node.left if node.left else node.right
                else:
                    node = node.right
            else:
                if feature_val < node.threshold:
                    node = node.left if node.left else node.right
                else:
                    node = node.right
        return node.val

    def _gini(self, y)
        """ See https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity """
        _, freq = np.unique(y, return_counts=True)
        rel_freq = freq / len(y)
        impurity = 1.0 - (rel_freq ** 2).sum()
        return impurity

    def _criterion(self, y, type="gini"):
        if type == "gini":
            return self._gini(y)
        else:
            raise NotImplementedError(f"Criterion {type} not implemented")

    def _try_split(self, X, y):
        # Empty set case
        if len(y) == 0:
            return None, None
        base_criterion = self._criterion(y, self.criterion)
        optimal_criterion = 1.0
        split_idx, split_thres = None, None
        # iterate thru each feature and compute optimal split criterion
        for idx in range(self.n_features):
            # sort features by value
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            # numerical features split
            if not isinstance(thresholds[0], str):
                for i in range(1, len(y)):
                    if thresholds[i] == thresholds[i-1]:
                        continue
                    thres = 0.5 * (thresholds[i] + thresholds[i-1]) # midpoint split
                    # weighted metric
                    criterion = (i / len(y)) * self._criterion(classes[:i], self.criterion) + (1. - i / len(y)) * self._criterion(classes[i:], self.criterion)
                    if criterion < optimal_criterion:
                        optimal_criterion, split_idx, split_thres = criterion, idx, thres
            # categorical features
            else:
                thres = None
                for i in range(1, len(y)):
                    if thresholds[i] == thresholds[i-1]:
                        continue
                    # split by prev value
                    thres = thresholds[i]
                    left_indices = X[:, idx] == thres # threshold by membership
                    criterion = (len(left_indices) / len(y)) * self._criterion(y[left_indices], self.criterion) + \
                        (1 - len(left_indices) / len(y)) * self._criterion(y[~left_indices], self.criterion)
                    if criterion < optimal_criterion:
                        optimal_criterion, split_idx, split_thres = criterion, idx, thres
                if thres and thres != thresholds[-1]:
                    left_indices = X[:, idx] == thresholds[-1]
                    criterion = (len(left_indices) / len(y)) * self._criterion(y[left_indices], self.criterion) + \
                        (1 - len(left_indices) / len(y)) * self._criterion(y[~left_indices], self.criterion)    
                    if criterion < optimal_criterion:
                        optimal_criterion, split_idx, split_thres = criterion, idx, thres                
        return split_idx, split_thres

    def _grow_tree(self, X, y, depth=0):
        # initialize current node with most frequent class 
        class_freq = [np.sum(y == self.class_val_for_id[i]) for i in range(self.n_classes)]
        node = Node(value=self.class_val_for_id[np.argmax(class_freq)])
        if depth < self.max_depth:
            # calculate split criteria
            split_feature_idx, split_thres = self._try_split(X, y)
            if split_feature_idx is not None:
                # try recursively split the tree after spliting data into the leaves
                left_indices = X[:, split_feature_idx] < split_thres
                right_indices = ~left_indices
                node = Node(
                    feature_idx=split_feature_idx,
                    threshold=split_thres,
                    left=self._grow_tree(X[left_indices], y[left_indices]),
                    right=self._grow_tree(X[right_indices], y[right_indices])
                )
        return node
