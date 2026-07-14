import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

CSV_PATH = "landmarks_data_twohands.csv"
MODEL_PATH = "gesture_model_twohands.pkl"
ENCODER_PATH = "label_encoder_twohands.pkl"

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"Data shape: {df.shape}")

    X = df.drop("label", axis=1).values
    y = df["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    )

    print("Training model...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved encoder to {ENCODER_PATH}")

if __name__ == "__main__":
    main()
