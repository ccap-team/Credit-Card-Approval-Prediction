# ============================================================
# Credit Card Approval Prediction
# Drop Duplicate Features / Records
# ============================================================

# Import Libraries
import pandas as pd

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
app = pd.read_csv("application_record.csv")

print("="*60)
print("Original Dataset Shape")
print("="*60)
print(app.shape)

# ------------------------------------------------------------
# Check Duplicate Rows
# ------------------------------------------------------------
duplicate_rows = app.duplicated().sum()

print("\nNumber of Duplicate Rows :", duplicate_rows)

# ------------------------------------------------------------
# Remove Duplicate Rows
# ------------------------------------------------------------
app = app.drop_duplicates(keep='first')

print("\nDataset Shape After Removing Duplicates")
print(app.shape)

# ------------------------------------------------------------
# Verify Again
# ------------------------------------------------------------
print("\nDuplicate Rows After Cleaning :", app.duplicated().sum())

# ------------------------------------------------------------
# Remove Duplicate Applicant IDs (if any)
# ------------------------------------------------------------
if "ID" in app.columns:
    app = app.drop_duplicates(subset=["ID"], keep="first")
    print("\nDataset Shape After Removing Duplicate IDs")
    print(app.shape)

# ------------------------------------------------------------
# Save Clean Dataset
# ------------------------------------------------------------
app.to_csv("application_record_no_duplicates.csv", index=False)

print("\nClean dataset saved successfully.")
print("File Name : application_record_no_duplicates.csv")

# ------------------------------------------------------------
# Display First 5 Rows
# ------------------------------------------------------------
print("\nFirst 5 Rows")
print(app.head())
