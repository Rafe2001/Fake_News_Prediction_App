from flask import Flask, request, render_template, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder="templates")
ps = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

# Load pickle model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

def predict(text):
    # Preprocess the input text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"\d+", " ", text)
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word not in stopwords_set]
    text = " ".join(text)

    # Vectorize the preprocessed text
    vect_text = vectorizer.transform([text]).toarray()

    # Make prediction
    prediction = "Fake" if model.predict(vect_text) == 0 else "Not Fake"
    return prediction

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def process_text():
    text = request.form['text']
    
    if not text:
        return render_template('index.html', text=text, result="Please enter text for prediction.")
    
    try:
        prediction = predict(text)
        return render_template('index.html', text=text, result=prediction)
    except Exception as e:
        return render_template('index.html', text=text, result=f"An error occurred: {str(e)}")

@app.route('/predict/', methods=['GET', 'POST'])
def api():
    text = request.args.get('text')
    
    if not text:
        return jsonify(error="Please provide text for prediction.")
    
    try:
        prediction = predict(text)
        return jsonify(prediction=prediction)
    except Exception as e:
        return jsonify(error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
