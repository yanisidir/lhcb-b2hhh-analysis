"""Signal/background dataset construction and ML helpers."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_FEATURES = [
    "B_VertexChi2",
    "B_FlightDistance",
    "H1_ProbK",
    "H2_ProbK",
    "H3_ProbK",
    "H1_ProbPi",
    "H2_ProbPi",
    "H3_ProbPi",
]


@dataclass(frozen=True)
class MassWindows:
    """Mass windows in MeV used to define proxy labels."""

    signal: tuple[float, float] = (5270.0, 5290.0)
    low_sideband: tuple[float, float] = (5200.0, 5250.0)
    high_sideband: tuple[float, float] = (5310.0, 5350.0)


def build_signal_background_dataset(
    df,
    mass_column: str = "B_mass",
    windows: MassWindows | None = None,
    features: list[str] | None = None,
):
    """Build a sideband-labelled signal/background table for ML studies.

    Events in the central B-mass window are labelled as signal-like. Events in
    the lower and upper sidebands are labelled as background-like. This creates
    proxy labels suitable for a didactic classifier, not a precision signal
    yield measurement.
    """
    import pandas as pd

    windows = windows or MassWindows()
    features = features or DEFAULT_FEATURES

    signal_mask = df[mass_column].between(*windows.signal)
    background_mask = df[mass_column].between(*windows.low_sideband) | df[mass_column].between(
        *windows.high_sideband
    )

    signal = df.loc[signal_mask, features + [mass_column]].copy()
    background = df.loc[background_mask, features + [mass_column]].copy()
    signal["label"] = 1
    background["label"] = 0

    ml_df = pd.concat([signal, background], ignore_index=True)
    ml_df = ml_df.dropna(subset=features + ["label"])
    return ml_df


def train_random_forest(
    ml_df,
    features: list[str] | None = None,
    target: str = "label",
    test_size: float = 0.30,
    random_state: int = 42,
):
    """Train a RandomForestClassifier and return the fitted objects."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    features = features or DEFAULT_FEATURES
    x = ml_df[features]
    y = ml_df[target]

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(x_train, y_train)
    return classifier, x_train, x_test, y_train, y_test


def compute_roc_curve(classifier, x_test, y_test):
    """Compute ROC points and AUC for a fitted classifier."""
    from sklearn.metrics import auc, roc_curve

    y_score = classifier.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc
