# ============================================================
# Credit Card Approval Prediction
# Handling Missing Values
# ============================================================

# Import Libraries
import pandas as pd

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
app = pd.read_csv("application_record.csv")

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

# ------------------------------------------------------------
# Dataset Shape
# ------------------------------------------------------------
print("\nOriginal Dataset Shape")
print(app.shape)

# ------------------------------------------------------------
# Missing Values Count
# ------------------------------------------------------------
print("\nMissing Values Count")
print(app.isnull().sum())

# ------------------------------------------------------------
# Missing Values Percentage
# ------------------------------------------------------------
print("\nMissing Values Percentage")
print((app.isnull().mean() * 100).round(2))

# ------------------------------------------------------------
# Display Columns Having Missing Values
# ------------------------------------------------------------
missing = app.isnull().sum()
missing = missing[missing > 0]

print("\nColumns with Missing Values")
print(missing)

# ------------------------------------------------------------
# Fill Missing Values
# (Occupation Type is categorical)
# ------------------------------------------------------------
if "OCCUPATION_TYPE" in app.columns:
    app["OCCUPATION_TYPE"].fillna("Unknown", inplace=True)

print("\nMissing Values After Filling")
print(app.isnull().sum())

# ------------------------------------------------------------
# Verify No Missing Values
# ------------------------------------------------------------
print("\nRemaining Missing Values")
print(app.isnull().sum().sum())

# ------------------------------------------------------------
# Save Clean Dataset
# ------------------------------------------------------------
app.to_csv("application_record_missing_handled.csv", index=False)

print("\nClean dataset saved successfully.")
print("File Name : application_record_missing_handled.csv")

# ------------------------------------------------------------
# Display First Five Rows
# ------------------------------------------------------------
print("\nFirst Five Rows")
print(app.head())
