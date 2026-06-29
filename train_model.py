import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Dataset/application_record.csv")

# Remove ID column
if "ID" in df.columns:
    df = df.drop("ID", axis=1)

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col].astype(str))

# Dummy target column (replace with real target if available)
if "TARGET" not in df.columns:
    df["TARGET"] = (df["AMT_INCOME_TOTAL"] > df["AMT_INCOME_TOTAL"].median()).astype(int)

X = df.drop("TARGET", axis=1)
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

# Save trained model
with open("Models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
