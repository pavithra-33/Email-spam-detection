# Email-spam-detection

📧 Email Spam Detector

A machine learning web application that classifies email messages as **Spam** or **Not Spam** using natural language processing and a Naive Bayes classifier. Built with Python, Scikit-learn, and Flask, this project demonstrates end-to-end ML workflow—from data preprocessing to model deployment in a browser-based interface.

🧠 Problem Statement

Spam emails reduce productivity and pose security risks. Can we automatically classify them using machine learning?


🎯 Objective

- Build a spam classifier using ML algorithms (Naive Bayes or SVM)
- Preprocess email text using NLP techniques
- Achieve 90%+ accuracy on test data
- Deploy a lightweight web app for real-time predictions

🚀 Features

- Text preprocessing: lowercasing, punctuation removal, stopword filtering
- TF-IDF vectorization for feature extraction
- Naive Bayes classification
- Model evaluation: accuracy, precision, recall
- Flask-based web interface for user input and prediction
- Ready for deployment on platforms like Render or Heroku

🧰 Tech Stack

| Layer         | Tools Used                     |
|--------------|---------------------------------|
| Language      | Python 3.10+                   |
| ML Framework  | Scikit-learn                   |
| NLP Toolkit   | NLTK                           |
| Web Framework | Flask                          |
| Frontend      | HTML, CSS                      |
| Deployment    | GitHub, Render/Heroku (optional) |

📁 Project Structure

email-spam-detector/
│
├── app.py                 # Flask web app
├── train_model.py         # Model training script
├── spam.csv               # Dataset (spam/ham labeled)
├── spam_model.pkl         # Saved ML model
├── vectorizer.pkl         # Saved TF-IDF vectorizer
├── templates/
│   └── index.html         # Web interface
├── .gitignore             # Git exclusions
└── README.md              # Project documentation

⚙️ Setup Instructions

1. Clone the Repository
```bash
git clone https://github.com/yourusername/email-spam-detector.git
cd email-spam-detector
```

2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

3. Install Dependencies
```bash
pip install pandas scikit-learn flask nltk
```

4. Download NLTK Stopwords
```python
import nltk
nltk.download('stopwords')
```

5. Train the Model
```bash
python train_model.py
```
6. Run the Web App
```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser.

---

Sample Inputs

Spam Example
```
Congratulations! You've won a $500 gift card. Click here to claim now!
```

Not Spam Example
```
Hi Pavithra, just confirming our meeting tomorrow at 3 PM.
```

---
📊 Model Performance

| Metric     | Score |
|------------|-------|
| Accuracy   | ~90%  |
| Precision  | High  |
| Recall     | High  |


