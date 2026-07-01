import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("merged_credit_card_dataset.csv", low_memory=False)

print("Dataset Loaded Successfully")
print("Dataset Shape :", df.shape)

# =====================================================
# Remove Missing STATUS
# =====================================================

df = df.dropna(subset=["STATUS"])

# =====================================================
# Create Target Variable
# =====================================================

good_status = ["C", "X"]

df["TARGET"] = df["STATUS"].apply(
    lambda x: 1 if str(x) in good_status else 0
)

# =====================================================
# Create User Friendly Features
# =====================================================

df["AGE"] = (df["DAYS_BIRTH"].abs() / 365).astype(int)

df["YEARS_EMPLOYED"] = (
    df["DAYS_EMPLOYED"]
    .replace(365243, 0)
    .abs() / 365
).astype(int)

# =====================================================
# Keep Only Features Used In Website
# =====================================================

selected_features = [

    "CODE_GENDER",

    "FLAG_OWN_CAR",

    "FLAG_OWN_REALTY",

    "CNT_CHILDREN",

    "AMT_INCOME_TOTAL",

    "NAME_INCOME_TYPE",

    "NAME_EDUCATION_TYPE",

    "NAME_FAMILY_STATUS",

    "NAME_HOUSING_TYPE",

    "OCCUPATION_TYPE",

    "CNT_FAM_MEMBERS",

    "AGE",

    "YEARS_EMPLOYED",

    "TARGET"

]

df = df[selected_features]

# =====================================================
# Fill Missing Values
# =====================================================

for col in df.columns:

    if df[col].dtype == "object":

        df[col] = df[col].fillna("Unknown")

    else:

        df[col] = df[col].fillna(df[col].median())

# =====================================================
# Encode Categorical Columns
# =====================================================

label_encoders = {}

for col in df.columns:

    if df[col].dtype == "object":

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        label_encoders[col] = encoder

# =====================================================
# Features & Target
# =====================================================

X = df.drop("TARGET", axis=1)

y = df["TARGET"]

feature_names = list(X.columns)

print("\nFeature Columns\n")
print(feature_names)

print("\nTarget Distribution\n")
print(y.value_counts())

# =====================================================
# Train Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

# =====================================================
# Train Random Forest
# =====================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=15,

    min_samples_split=5,

    random_state=42,

    n_jobs=-1

)

print("\nTraining Model...\n")

model.fit(X_train, y_train)

# =====================================================
# Evaluate Model
# =====================================================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report\n")

print(classification_report(y_test, prediction))

# =====================================================
# Save Model Files
# =====================================================

os.makedirs("Models", exist_ok=True)

with open("Models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("Models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("Models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\n===================================")
print("Model Saved Successfully!")
print("===================================")
print("Model            : Models/model.pkl")
print("Label Encoders   : Models/label_encoders.pkl")
print("Feature Names    : Models/feature_names.pkl")
