import pandas as pd

# Load the datasets
application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

# Display first 5 rows
print("Application Dataset")
display(application.head())

print("Credit Record Dataset")
display(credit.head())

# Display dataset information
print("\nApplication Dataset Info")
application.info()

print("\nCredit Record Dataset Info")
credit.info()

# Display shapes
print("\nApplication Dataset Shape:", application.shape)
print("Credit Record Dataset Shape:", credit.shape)
