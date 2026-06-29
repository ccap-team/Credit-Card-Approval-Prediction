# ============================================================
# Credit Card Approval Prediction
# Multivariate Analysis
# ============================================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
app = pd.read_csv("application_record.csv")

print("Dataset Loaded Successfully")
print("Shape :", app.shape)

# ------------------------------------------------------------
# Create Age Column
# ------------------------------------------------------------
app["Age"] = (-app["DAYS_BIRTH"]) / 365

# ------------------------------------------------------------
# Select Numerical Columns
# ------------------------------------------------------------
numerical_columns = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "DAYS_EMPLOYED",
    "Age",
    "CNT_FAM_MEMBERS"
]

corr = app[numerical_columns].corr()

print("\nCorrelation Matrix")
print(corr)

# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------
plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------------------------------
# Income vs Age
# ------------------------------------------------------------
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=app,
    x="Age",
    y="AMT_INCOME_TOTAL"
)

plt.title("Age vs Annual Income")
plt.xlabel("Age")
plt.ylabel("Annual Income")

plt.show()

# ------------------------------------------------------------
# Income by Education
# ------------------------------------------------------------
plt.figure(figsize=(10,6))

sns.boxplot(
    data=app,
    x="NAME_EDUCATION_TYPE",
    y="AMT_INCOME_TOTAL"
)

plt.xticks(rotation=20)

plt.title("Income by Education Level")

plt.show()

# ------------------------------------------------------------
# Income by Gender
# ------------------------------------------------------------
plt.figure(figsize=(6,5))

sns.boxplot(
    data=app,
    x="CODE_GENDER",
    y="AMT_INCOME_TOTAL"
)

plt.title("Income Distribution by Gender")

plt.show()

# ------------------------------------------------------------
# Family Members vs Income
# ------------------------------------------------------------
plt.figure(figsize=(8,5))

sns.scatterplot(
    data=app,
    x="CNT_FAM_MEMBERS",
    y="AMT_INCOME_TOTAL",
    hue="CODE_GENDER"
)

plt.title("Family Members vs Income")

plt.show()

# ------------------------------------------------------------
# Income Type vs Housing Type
# ------------------------------------------------------------
plt.figure(figsize=(12,6))

sns.countplot(
    data=app,
    x="NAME_INCOME_TYPE",
    hue="FLAG_OWN_REALTY"
)

plt.xticks(rotation=25)

plt.title("Income Type vs House Ownership")

plt.show()

# ------------------------------------------------------------
# Occupation vs Education
# ------------------------------------------------------------
cross = pd.crosstab(
    app["NAME_EDUCATION_TYPE"],
    app["FLAG_OWN_CAR"]
)

print("\nEducation vs Own Car")
print(cross)

plt.figure(figsize=(10,6))

sns.heatmap(
    cross,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Education vs Car Ownership")

plt.show()

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\nMultivariate Analysis Completed Successfully.")
