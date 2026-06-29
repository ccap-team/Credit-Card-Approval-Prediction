# ============================================================
# Credit Card Approval Prediction
# Handling Categorical Values
# ============================================================

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("application_record.csv")

print("Dataset Shape:", df.shape)

# ------------------------------------------------------------
# Fill Missing Values
# ------------------------------------------------------------
df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

# ------------------------------------------------------------
# Encode Categorical Columns
# ------------------------------------------------------------
label = LabelEncoder()

categorical_columns = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

for column in categorical_columns:
    df[column] = label.fit_transform(df[column])

# ------------------------------------------------------------
# Display Encoded Data
# ------------------------------------------------------------
print("\nEncoded Dataset")
print(df.head())

print("\nData Types")
print(df.dtypes)

# ------------------------------------------------------------
# Save Encoded Dataset
# ------------------------------------------------------------
df.to_csv("encoded_application_record.csv", index=False)

print("\nCategorical value encoding completed successfully.")
