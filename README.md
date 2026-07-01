# 💳 Credit Card Approval Prediction

A Machine Learning-based web application that predicts whether a credit card application is likely to be **Approved** or **Rejected** based on applicant information. The project demonstrates the complete machine learning workflow, from data preprocessing and model training to deployment using Flask.

---

## 🚀 Features

- Predicts credit card approval status
- User-friendly web interface built with Flask
- Data preprocessing and feature engineering
- Multiple machine learning algorithms implemented and evaluated
- Random Forest selected as the final prediction model
- Live deployed application
- Responsive UI for easy interaction

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- HTML5
- CSS3

---

## 📂 Project Structure

```
Credit-Card-Approval-Prediction/
│
├── Dataset/
├── Documentation/
├── Models/
│   ├── model.pkl
│   ├── feature_names.pkl
│   └── label_encoders.pkl
│
├── Notebooks/
├── Screenshots/
├── Static/
│   └── style.css
│
├── Templates/
│   ├── home.html
│   ├── index.html
│   └── result.html
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Machine Learning Workflow

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Categorical Encoding
- Model Training
- Model Evaluation
- Flask Web Application Development
- Deployment

---

## 🤖 Machine Learning Models

The following classification algorithms were implemented and evaluated:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

After comparing their performance, the **Random Forest Classifier** was selected as the final model for deployment.

---

## 📸 Screenshots

The project includes screenshots demonstrating:

- Home Page
- Prediction Form
- Credit Card Approved Result
- Credit Card Rejected Result
- Model Training Output
- Classification Report
- Project Structure
- Flask Application Running

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone <repository-url>
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Train the model

```bash
python train_model.py
```

4. Start the Flask application

```bash
python app.py
```

5. Open your browser

```
http://127.0.0.1:5000
```

---

## 📈 Results

- Successfully trained a Machine Learning model for credit card approval prediction.
- Developed an interactive Flask web application.
- Deployed the project for live demonstration.
- Demonstrated the complete Machine Learning pipeline from preprocessing to deployment.

---

## 👨‍💻 Team

Developed as part of the **SkillWallet Artificial Intelligence & Machine Learning Internship**.

---

## 📄 License

This project was developed for educational purposes as part of the SkillWallet Internship Program.

---

## 🎯 Future Improvements

- Improve model accuracy using advanced feature engineering.
- Add user authentication.
- Connect to a real-time banking database.
- Deploy using cloud platforms for scalability.
- Enhance the UI with additional analytics and visualizations.
