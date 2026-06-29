# ============================================================
# Credit Card Approval Prediction
# Feature Engineering
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Load Datasets
# ------------------------------------------------------------
application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

# ------------------------------------------------------------
# Handle Missing Values
# ------------------------------------------------------------
application["OCCUPATION_TYPE"] = application["OCCUPATION_TYPE"].fillna("Unknown")

# ------------------------------------------------------------
# Convert Negative Days into Positive Values
# ------------------------------------------------------------
application["DAYS_BIRTH"] = application["DAYS_BIRTH"].abs()
application["DAYS_EMPLOYED"] = application["DAYS_EMPLOYED"].abs()

# ------------------------------------------------------------
# Create Age Feature
# ------------------------------------------------------------
application["AGE"] = (application["DAYS_BIRTH"] / 365).astype(int)

# ------------------------------------------------------------
# Create Family Size Feature
# ------------------------------------------------------------
application["FAMILY_SIZE"] = (
    application["CNT_CHILDREN"] +
    application["CNT_FAM_MEMBERS"]
)

# ------------------------------------------------------------
# Convert STATUS to Binary Target
# 1 = Approved (Good History)
# 0 = Not Approved (Poor History)
# ------------------------------------------------------------
def status_to_binary(status):
    if status in ['0', 'X', 'C']:
        return 1
    else:
        return 0

credit["STATUS_BIN"] = credit["STATUS"].apply(status_to_binary)

print("Binary Status Counts")
print(credit["STATUS_BIN"].value_counts())

# ------------------------------------------------------------
# Merge Application and Credit Records
# ------------------------------------------------------------
final_df = application.merge(
    credit,
    how="left",
    on="ID"
)

print("\nMerged Dataset Shape:")
print(final_df.shape)

# ------------------------------------------------------------
# Missing Values After Merge
# ------------------------------------------------------------
print("\nMissing Values")
print(final_df.isnull().sum().sort_values(ascending=False).head(10))

# ------------------------------------------------------------
# Preview
# ------------------------------------------------------------
print("\nFirst Five Rows")
print(final_df.head())

# ------------------------------------------------------------
# Save Engineered Dataset
# ------------------------------------------------------------
final_df.to_csv("feature_engineered_dataset.csv", index=False)

print("\nFeature engineering completed successfully.")
