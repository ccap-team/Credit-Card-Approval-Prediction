# ============================================================
# Credit Card Approval Prediction
# Data Cleaning and Merging
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Load Datasets
# ------------------------------------------------------------
application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

print("="*60)
print("Application Dataset Shape :", application.shape)
print("Credit Dataset Shape :", credit.shape)

# ------------------------------------------------------------
# Remove Duplicate Records
# ------------------------------------------------------------
application = application.drop_duplicates()
credit = credit.drop_duplicates()

print("\nAfter Removing Duplicates")
print("Application :", application.shape)
print("Credit :", credit.shape)

# ------------------------------------------------------------
# Handle Missing Values
# ------------------------------------------------------------
application["OCCUPATION_TYPE"] = application["OCCUPATION_TYPE"].fillna("Unknown")

print("\nMissing Values")
print(application.isnull().sum())

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
# Merge Datasets using Applicant ID
# ------------------------------------------------------------
merged = pd.merge(
    application,
    credit,
    on="ID",
    how="left"
)

print("\nMerged Dataset Shape")
print(merged.shape)

# ------------------------------------------------------------
# Check Missing Values After Merge
# ------------------------------------------------------------
print("\nMissing Values After Merge")
print(merged.isnull().sum())

# ------------------------------------------------------------
# Save Final Dataset
# ------------------------------------------------------------
merged.to_csv("merged_credit_card_dataset.csv", index=False)

print("\nMerged dataset saved successfully.")

# ------------------------------------------------------------
# Preview
# ------------------------------------------------------------
print("\nFirst Five Rows")
print(merged.head())
