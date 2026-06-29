# ============================================================
# Credit Card Approval Prediction - Univariate Analysis
# ============================================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Display graphs properly
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# ============================================================
# Read Dataset
# ============================================================

app = pd.read_csv("application_record.csv")

print("="*60)
print("First 5 Records")
print("="*60)
print(app.head())

print("\n")

print("="*60)
print("Dataset Information")
print("="*60)
print(app.info())

print("\n")

print("="*60)
print("Dataset Shape")
print("="*60)
print(app.shape)

print("\n")

print("="*60)
print("Column Names")
print("="*60)
print(app.columns)

# ============================================================
# Univariate Analysis - Gender
# ============================================================

print("\nGender Distribution")
print(app["CODE_GENDER"].value_counts())

plt.figure()
sns.countplot(x="CODE_GENDER", data=app)
plt.title("Gender Distribution")
plt.show()

# ============================================================
# Income Type
# ============================================================

print("\nIncome Type Distribution")
print(app["NAME_INCOME_TYPE"].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(y="NAME_INCOME_TYPE", data=app)
plt.title("Income Type Distribution")
plt.show()

# ============================================================
# Education Level
# ============================================================

print("\nEducation Level")
print(app["NAME_EDUCATION_TYPE"].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(y="NAME_EDUCATION_TYPE", data=app)
plt.title("Education Level")
plt.show()

# ============================================================
# Family Status
# ============================================================

print("\nFamily Status")
print(app["NAME_FAMILY_STATUS"].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(y="NAME_FAMILY_STATUS", data=app)
plt.title("Family Status")
plt.show()

# ============================================================
# Housing Type
# ============================================================

print("\nHousing Type")
print(app["NAME_HOUSING_TYPE"].value_counts())

plt.figure(figsize=(10,5))
sns.countplot(y="NAME_HOUSING_TYPE", data=app)
plt.title("Housing Type")
plt.show()

# ============================================================
# Occupation Type
# ============================================================

print("\nOccupation Type")
print(app["OCCUPATION_TYPE"].value_counts(dropna=False))

plt.figure(figsize=(12,8))
sns.countplot(
    y="OCCUPATION_TYPE",
    data=app,
    order=app["OCCUPATION_TYPE"].value_counts().index
)
plt.title("Occupation Type")
plt.show()

# ============================================================
# Own Car
# ============================================================

print("\nOwn Car")
print(app["FLAG_OWN_CAR"].value_counts())

plt.figure()
sns.countplot(x="FLAG_OWN_CAR", data=app)
plt.title("Own Car")
plt.show()

# ============================================================
# Own House
# ============================================================

print("\nOwn Realty")
print(app["FLAG_OWN_REALTY"].value_counts())

plt.figure()
sns.countplot(x="FLAG_OWN_REALTY", data=app)
plt.title("Own House")
plt.show()

# ============================================================
# Income Distribution
# ============================================================

plt.figure(figsize=(10,5))
sns.histplot(app["AMT_INCOME_TOTAL"], bins=30, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Income")
plt.show()

# ============================================================
# Age Distribution
# DAYS_BIRTH is negative, convert to years
# ============================================================

app["Age"] = (-app["DAYS_BIRTH"]) / 365

plt.figure(figsize=(10,5))
sns.histplot(app["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

# ============================================================
# Number of Children
# ============================================================

plt.figure(figsize=(8,5))
sns.countplot(x="CNT_CHILDREN", data=app)
plt.title("Number of Children")
plt.show()

# ============================================================
# Family Members
# ============================================================

plt.figure(figsize=(8,5))
sns.countplot(x="CNT_FAM_MEMBERS", data=app)
plt.title("Family Members")
plt.show()

print("\n")
print("="*60)
print("Univariate Analysis Completed Successfully")
print("="*60)
