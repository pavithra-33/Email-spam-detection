from flask import Flask, request, render_template
import pickle

# Load model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
