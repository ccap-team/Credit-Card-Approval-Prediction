# ============================================================
# Credit Card Approval Prediction
# Decision Tree Model
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

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
# Encode Categorical Columns
# ------------------------------------------------------------
encoder = LabelEncoder()

for col in application.select_dtypes(include="object").columns:
    application[col] = encoder.fit_transform(application[col].astype(str))

# ------------------------------------------------------------
# Convert Credit Status to Binary
# ------------------------------------------------------------
credit["STATUS_BIN"] = credit["STATUS"].apply(
    lambda x: 1 if x in ["0", "X", "C"] else 0
)

credit = credit.groupby("ID")["STATUS_BIN"].max().reset_index()

# ------------------------------------------------------------
# Merge Datasets
# ------------------------------------------------------------
df = application.merge(credit, on="ID", how="inner")

# ------------------------------------------------------------
# Features & Target
# ------------------------------------------------------------
X = df.drop(["ID", "STATUS_BIN"], axis=1)
y = df["STATUS_BIN"]

# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------------------------
# Train Decision Tree Model
# ------------------------------------------------------------
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=6
)

dt_model.fit(X_train, y_train)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
y_pred = dt_model.predict(X_test)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Accuracy :", accuracy)

# ------------------------------------------------------------
# Classification Report
# ------------------------------------------------------------
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------
importance = pd.Series(
    dt_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.head(10).plot(kind="bar")
plt.title("Top 10 Important Features")
plt.ylabel("Importance")
plt.show()
