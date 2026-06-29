# ============================================================
# Credit Card Approval Prediction
# Random Forest Model
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
# Fill Missing Values
# ------------------------------------------------------------
application["OCCUPATION_TYPE"] = application["OCCUPATION_TYPE"].fillna("Unknown")

# ------------------------------------------------------------
# Encode Categorical Columns
# ------------------------------------------------------------
encoder = LabelEncoder()

for column in application.select_dtypes(include="object").columns:
    application[column] = encoder.fit_transform(application[column].astype(str))

# ------------------------------------------------------------
# Convert STATUS to Binary Target
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
# Features and Target
# ------------------------------------------------------------
X = df.drop(["ID", "STATUS_BIN"], axis=1)
y = df["STATUS_BIN"]

# ------------------------------------------------------------
# Split Dataset
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ------------------------------------------------------------
# Train Random Forest Model
# ------------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------
y_pred = rf_model.predict(X_test)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Accuracy :", accuracy)

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
    cmap="Greens"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ------------------------------------------------------------
# Feature Importance
# ------------------------------------------------------------
importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
importance.head(10).plot(kind="bar")
plt.title("Top 10 Important Features")
plt.ylabel("Importance Score")
plt.show()
