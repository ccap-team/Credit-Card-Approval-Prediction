# ============================================================
# Credit Card Approval Prediction
# Logistic Regression Model
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# Load Dataset
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
categorical_columns = application.select_dtypes(include="object").columns

encoder = LabelEncoder()
for col in categorical_columns:
    application[col] = encoder.fit_transform(application[col].astype(str))

# ------------------------------------------------------------
# Create Binary Target from Credit History
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
    test_size=0.2,
    random_state=42
)

# ------------------------------------------------------------
# Train Logistic Regression
# ------------------------------------------------------------
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

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
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues")

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
