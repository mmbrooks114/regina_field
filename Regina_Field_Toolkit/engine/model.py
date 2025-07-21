from sklearn.ensemble import RandomForestClassifier

def train_boundary_model(df, label_col="IsPrime"):
    """
    Train a simple random forest model to classify prime-like vs non-prime-like
    based on current structural features.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    features = ["MotifSum", "Entropy", "HilbertMag", "BoundaryTransitionIndex"]
    df_clean = df.dropna(subset=features + [label_col])
    X = df_clean[features]
    y = df_clean[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return clf

def apply_boundary_model(df, model):
    """
    Apply trained model to calculate updated boundary likelihoods.
    """
    features = ["MotifSum", "Entropy", "HilbertMag", "BoundaryTransitionIndex"]
    df["BoundaryScore"] = model.predict_proba(df[features])[:, 1]
    return df