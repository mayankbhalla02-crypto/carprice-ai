"""Price-band lookup for the CarDekho dataset (fail-safe version).

Bands and prices come straight from Cardekho.csv, so the page works even if
the saved model file cannot be loaded. The RandomForest is only an optional
extra check.
"""
import difflib
import threading
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cardekho.csv"
MODEL_PATH = BASE_DIR / "car_model_rf.pkl"

BAND_NAMES = ["Budget", "Mid-range", "Premium"]
RUPEE = "\u20b9"


class UnknownCar(ValueError):
    def __init__(self, name, suggestions):
        super().__init__(name)
        self.name = name
        self.suggestions = suggestions


def _clean(value):
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = str(value).strip()
    return "" if text in ("None", "nan", "NaN") else text


def _money(value):
    lakh = value / 1e5
    if lakh >= 100:
        crore = f"{lakh / 100:.2f}".rstrip("0").rstrip(".")
        return f"{RUPEE}{crore} crore"
    return f"{RUPEE}{lakh:.0f} lakh"


def _display_price(text):
    return text.replace("Rs.", RUPEE).replace("Rs", RUPEE).strip()


class _State:
    pass


_state = None
_lock = threading.Lock()


def get_state():
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = _build_state()
    return _state


def _build_state():
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    df["model"] = df["model"].astype(str).str.strip()
    df["max_price"] = pd.to_numeric(df["max_price"], errors="coerce")
    df = df.dropna(subset=["max_price"]).reset_index(drop=True)

    edges = [float(e) for e in pd.qcut(df["max_price"], q=3, retbins=True)[1]]
    low, high = _money(edges[1]), _money(edges[2])

    st = _State()
    st.edges = edges
    st.ranges = [f"Up to {low}", f"{low} to {high}", f"Above {high}"]
    st.cars = []
    for _, r in df.iterrows():
        st.cars.append(
            {
                "model": r["model"],
                "fuel": _clean(r["fuel"]),
                "type": _clean(r["Type"]),
                "price": _clean(r["price"]).rstrip("*").strip(),
                "max_price": float(r["max_price"]),
                "url": _clean(r["url"]),
            }
        )
    st.names = [c["model"] for c in st.cars]
    st.lookup = {}
    for car in st.cars:
        st.lookup.setdefault(car["model"].lower(), car)

    # Optional: the saved RandomForest (never required)
    st.model = None
    st.columns = []
    try:
        import warnings

        import joblib

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = joblib.load(MODEL_PATH)
        if isinstance(obj, tuple):
            obj = obj[0]
        st.columns = list(obj.feature_names_in_)
        st.model = obj
    except Exception as exc:
        print("Saved model not used:", repr(exc), flush=True)
    return st


def known_cars():
    return get_state().cars


def band_info():
    st = get_state()
    return [{"label": BAND_NAMES[i], "range": st.ranges[i]} for i in range(3)]


def _model_band(st, car):
    if st.model is None:
        return None
    try:
        row = pd.DataFrame(0, index=[0], columns=st.columns, dtype=int)
        for col in (f"model_{car['model']}", f"fuel_{car['fuel']}", f"Type_{car['type']}"):
            if col in row.columns:
                row.at[0, col] = 1
        return int(st.model.predict(row)[0])
    except Exception as exc:
        print("Model prediction skipped:", repr(exc), flush=True)
        return None


def predict_price(payload):
    st = get_state()
    name = str(payload.get("model") or "").strip()
    car = st.lookup.get(name.lower())
    if car is None and name:
        hits = [c for c in st.cars if name.lower() in c["model"].lower()]
        if len(hits) == 1:
            car = hits[0]
    if car is None:
        raise UnknownCar(name, difflib.get_close_matches(name, st.names, n=5, cutoff=0.4))

    mp = car["max_price"]
    if mp <= st.edges[1]:
        category = 0
    elif mp <= st.edges[2]:
        category = 1
    else:
        category = 2

    model_category = _model_band(st, car)
    return {
        "price_category": category,
        "category_label": BAND_NAMES[category],
        "price_range": st.ranges[category],
        "estimated_range": f"{st.ranges[category]} ({BAND_NAMES[category]})",
        "model_category_label": BAND_NAMES[model_category] if model_category is not None else None,
        "model_agrees": (model_category == category) if model_category is not None else None,
        "matched_model": car["model"],
        "fuel": car["fuel"] or "Not listed",
        "transmission": car["type"] or "Not listed",
        "listed_price": _display_price(car["price"]),
        "url": car["url"],
        "bands": band_info(),
    }
