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
    st.edges = [float(e) for e in edges]
    st.ranges = _band_ranges(edges)
    st.cars = [
        {
            "model": r.model,
            "fuel": _clean(r.fuel),
            "type": _clean(r.Type),
            "price": _clean(r.price).rstrip("*").strip(),
            "max_price": float(r.max_price),
            "url": _clean(r.url),
        }
        for r in df.itertuples()
    ]
    st.names = [c["model"] for c in st.cars]
    st.lookup = {}
    for car in st.cars:
        st.lookup.setdefault(car["model"].lower(), car)
    return st


def _display_price(text):
    return text.replace("Rs.", RUPEE).replace("Rs", RUPEE).strip()


def predict_price(payload):
    st = get_state()
    name = str(payload.get("model") or "").strip()
    car = st.lookup.get(name.lower())
    if car is None:
        raise UnknownCar(name, difflib.get_close_matches(name, st.names, n=5, cutoff=0.5))

    # Use the car's OWN fuel/type (as in the training data), so the model sees
    # exactly the row it was trained on. No fallback to Petrol/Manual.
    fuel, transmission = car["fuel"], car["type"]
    row = pd.DataFrame(0, index=[0], columns=st.columns, dtype=int)
    for col in (f"model_{car['model']}", f"fuel_{fuel}", f"Type_{transmission}"):
        if col in row.columns:
            row.at[0, col] = 1
    model_category = int(st.model.predict(row)[0])

    # Band taken directly from CarDekho's own listed (highest) price
    mp = car["max_price"]
    category = 0 if mp <= st.edges[1] else (1 if mp <= st.edges[2] else 2)

    return {
        "price_category": category,
        "category_label": BAND_NAMES[category],
        "price_range": st.ranges[category],
        "estimated_range": f"{st.ranges[category]} ({BAND_NAMES[category]})",
        "model_category_label": BAND_NAMES[model_category],
        "model_agrees": model_category == category,
        "matched_model": car["model"],
        "fuel": fuel or "Not listed",
        "transmission": transmission or "Not listed",
        "listed_price": _display_price(car["price"]),
        "url": car["url"],
        "bands": band_info(),
    }
