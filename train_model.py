import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

print("Loading data...")
df = pd.read_csv("gesture_data.csv")

print(f"Total samples: {len(df)}")
print(f"Gestures: {df['label'].value_counts().to_dict()}")

X = df.drop("label", axis=1).values
y = df["label"].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Classes: {list(encoder.classes_)}")

print("\nTraining neural network...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=True
    ))
])

model.fit(X_train, y_train)

print("\nEvaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.1f}%")
print("\nDetailed results:")
print(classification_report(y_test, y_pred,
      target_names=encoder.classes_))

with open("gesture_model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoder": encoder}, f)

print("\nModel saved to gesture_model.pkl")