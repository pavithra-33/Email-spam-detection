import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not already present
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load and clean dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['label', 'text']]
df.columns = ['label', 'text']
df = df.dropna(subset=['label', 'text'])

# Map labels and filter valid rows
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna(subset=['label_num'])

# Preprocess text
df['text'] = df['text'].apply(preprocess)

# Vectorize text
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])
y = df['label_num']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
