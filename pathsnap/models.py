"""The candidate classifiers compared in the experiments (one per algorithm family)."""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

# name -> (family, factory, max training rows or None, config string for the report)
MODELS = {
    "LogisticRegression": ("Linear / probabilistic",
                           lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                           None, "StandardScaler, C=1.0, L2, lbfgs"),
    "GaussianNB": ("Probabilistic (Bayesian)", lambda: GaussianNB(), None, "default var_smoothing=1e-9"),
    "KNN": ("Instance-based",
            lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
            150_000, "StandardScaler, k=25, distance-weighted, 150k-row training sample"),
    "DecisionTree": ("Tree-based",
                     lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED),
                     None, "max_depth=12, min_samples_leaf=20"),
    "RandomForest": ("Ensemble (bagging)",
                     lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features="sqrt",
                                                    n_jobs=-1, random_state=SEED),
                     None, "150 trees, min_samples_leaf=5, max_features=sqrt"),
    "HistGradientBoosting": ("Boosting",
                             lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200, max_leaf_nodes=31,
                                                                    random_state=SEED),
                             None, "lr=0.1, 200 iterations, 31 leaves"),
    "MLP": ("Neural network",
            lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                                  batch_size=1024, learning_rate_init=1e-3,
                                                                  max_iter=40, early_stopping=True,
                                                                  validation_fraction=0.1, n_iter_no_change=5,
                                                                  random_state=SEED)),
            None, "StandardScaler, layers (64,32), ReLU, Adam lr=1e-3, batch 1024, early stopping"),
}
