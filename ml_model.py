"""Price-band model for the CarDekho dataset.

Predictions come from the trained RandomForest in car_model_rf.pkl.

If that file cannot be used on this machine (for example a different
scikit-learn version) or no longer agrees with Cardekho.csv, the same model is
rebuilt from the CSV automatically and saved as car_model_rf.local.pkl. The
original .pkl is never modified.

The model was trained on car names, so it can rate the cars listed in the CSV.
Names it has never seen are rejected instead of being guessed.
"""
import difflib
import logging
import threading
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cardekho.csv"
MODEL_PATH = BASE_DIR / "car_model_rf.pkl"
LOCAL_MODEL_PATH = BASE_DIR / "car_model_rf.local.pkl"

FUEL_OPTIONS = ["Petrol", "Petrol/CNG"]
TYPE_OPTIONS = ["Manual", "Manual/Automatic"]
BAND_NAMES = ["Budget", "Mid-range", "Premium"]
MIN_AGREEMENT = 0.95  # a usable model must reproduce its own training labels

RUPEE = "\u20b9"
log = logging.getLogger("carai.model")


class UnknownCar(ValueError):
    """Raised when the requested car name is not in the training data."""

    def __init__(self, name, suggestions):
        super().__init__(name)
        self.name = name
        self.suggestions = suggestions


# --------------------------------------------------------------------------
# Data and features
# --------------------------------------------------------------------------
def _read_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cardekho.csv is missing. Keep it in the same folder as app.py."
        )
    df = pd.read_csv(DATA_PATH)
    df["model"] = df["model"].astype(str).str.strip()
    return df


def _design_matrix(df):
    """One-hot features, built the same way the saved model was trained."""
    return pd.get_dummies(df[["model", "fuel", "Type"]], drop_first=True).astype(int)


def _labels(df):
    y, edges = pd.qcut(df["max_price"], q=3, labels=[0, 1, 2], retbins=True)
    return y.astype(int).to_numpy(), edges


def _money(value):
    lakh = value / 1e5
    if lakh >= 100:
        crore = f"{lakh / 100:.2f}".rstrip("0").rstrip(".")
        return f"{RUPEE}{crore} crore"
    return f"{RUPEE}{lakh:.0f} lakh"


def _band_ranges(edges):
    low, high = _money(edges[1]), _money(edges[2])
    return [f"Up to {low}", f"{low} to {high}", f"Above {high}"]


def _clean(value):
    return "" if pd.isna(value) else str(value).strip()


# --------------------------------------------------------------------------
# Loading / training
# --------------------------------------------------------------------------
def _try_load(path, X, y):
    """Return a usable model from `path`, or None."""
    if not path.exists():
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(path)
            if isinstance(obj, tuple):  # older (model, columns) format
                obj = obj[0]
            columns = list(obj.feature_names_in_)
            if set(columns) != set(X.columns):  # cars added/removed in the CSV
                log.warning("%s was trained on different data than Cardekho.csv", path.name)
                return None
            agreement = float(np.mean(obj.predict(X[columns]) == y))
    except Exception as exc:  # unreadable, wrong type, version clash, ...
        log.warning("Could not use %s (%s)", path.name, exc)
        return None
    if agreement < MIN_AGREEMENT:
        log.warning("%s disagrees with Cardekho.csv (%.0f%%)", path.name, agreement * 100)
        return None
    return obj


def _train(X, y):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model


class _State:
    pass


_state = None
_lock = threading.Lock()


def get_state():
    """Load everything once, on first use."""
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = _build_state()
    return _state


def _build_state():
    df = _read_data()
    y, edges = _labels(df)
    X = _design_matrix(df)

    model, source = None, None
    for path in (MODEL_PATH, LOCAL_MODEL_PATH):
        model = _try_load(path, X, y)
        if model is not None:
            source = path.name
            break
    if model is None:
        log.warning("Rebuilding the model from Cardekho.csv ...")
        model = _train(X, y)
        try:
            joblib.dump(model, LOCAL_MODEL_PATH)
        except OSError as exc:
            log.warning("Could not save %s (%s)", LOCAL_MODEL_PATH.name, exc)
        source = "rebuilt from Cardekho.csv"

    st = _State()
    st.model = model
    st.columns = list(model.feature_names_in_)
    st.source = source
    st.ranges = _band_ranges(edges)
    st.cars = [
        {
            "model": r.model,
            "fuel": _clean(r.fuel),
            "type": _clean(r.Type),
            "price": _clean(r.price).rstrip("*").strip(),
            "url": _clean(r.url),
        }
        for r in df.itertuples()
    ]
    st.names = [c["model"] for c in st.cars]
    st.lookup = {}
    for car in st.cars:
        st.lookup.setdefault(car["model"].lower(), car)
    return st


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def known_cars():
    return get_state().cars


def band_info():
    st = get_state()
    return [{"label": BAND_NAMES[i], "range": st.ranges[i]} for i in range(3)]


def predict_price(payload):
    st = get_state()
    name = str(payload.get("model") or "").strip()
    car = st.lookup.get(name.lower())
    if car is None:
        raise UnknownCar(name, difflib.get_close_matches(name, st.names, n=5, cutoff=0.5))

    fuel = payload.get("fuel") or car["fuel"] or FUEL_OPTIONS[0]
    transmission = payload.get("Type") or car["type"] or TYPE_OPTIONS[0]

    row = pd.DataFrame(0, index=[0], columns=st.columns, dtype=int)
    for col in (f"model_{car['model']}", f"fuel_{fuel}", f"Type_{transmission}"):
        if col in row.columns:  # the baseline category has no column (all zeros)
            row.at[0, col] = 1
    category = int(st.model.predict(row)[0])

    return {
        "price_category": category,
        "category_label": BAND_NAMES[category],
        "price_range": st.ranges[category],
        "estimated_range": f"{st.ranges[category]} ({BAND_NAMES[category]})",
        "matched_model": car["model"],
        "fuel": fuel,
        "transmission": transmission,
        "listed_price": car["price"],
        "url": car["url"],
        "bands": band_info(),
    }
