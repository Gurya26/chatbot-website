from flask import Flask, render_template, request, jsonify # type: ignore
import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os

base_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(base_dir, "templates"),
            static_folder=os.path.join(base_dir, "static"))

# Load data
data = pd.read_csv("faq.csv")
questions = data['question'].tolist()
answers = data['answer'].tolist()

# Vectorizepython app.py
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(questions)

def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    max_score = similarity.max()
    index = similarity.argmax()

    if max_score < 0.3:
        return "Sorry 😔 I didn't understand that. Try asking something else."

    return answers[index]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)