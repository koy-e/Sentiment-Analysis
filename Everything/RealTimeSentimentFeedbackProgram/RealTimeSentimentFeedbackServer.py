import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from http.server import SimpleHTTPRequestHandler
from http.server import HTTPServer
from http import HTTPStatus
from http.server import test
from urllib.parse import parse_qs
import string
import re
import joblib
import certifi
import os
import ssl
import json

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

nltk_data_path = "./nltkDependencies"
nltk.data.path.append(nltk_data_path)

# Set the CA certificates path
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Download the corpora if not already present in the folder
if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_path)

if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)

if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'wordnet')):
    nltk.download('wordnet', download_dir=nltk_data_path)


# # Preprocessing function
stop_words = set(stopwords.words('english'))


def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    words = [words[i] + '_' + words[i + 1] if words[i] == 'not' and i +
             1 < len(words) else words[i] for i in range(len(words))]

    lemmatized = [lemmatizer.lemmatize(
        word) for word in words if word.isalpha() and word not in stop_words]

    return ' '.join(lemmatized)


def predict_sentiment(review, model, vectorizer):
    # Preprocess and predict
    processed_review = preprocess_text(review)
    vectorized_review = vectorizer.transform([processed_review])
    probabilities = model.predict_proba(vectorized_review)
    # Scale probability to be between -1 and 1
    score = 2 * probabilities[0][1] - 1
    return score


def main():
    # Load the model and vectorizer
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')

    while True:
        review_text = input("Enter a review (or type 'exit' to quit): ")
        if review_text.lower() == 'exit':
            break
        sentiment_score = predict_sentiment(review_text, model, vectorizer)
        print(
            f"Sentiment score ranging from -1 (very negative) to 1 (very positive):{sentiment_score:.2f}")


class SentimentAnalysisHandler(SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Origin, Content-Type, Accept')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            # Parse JSON data from the request body
            post_params = json.loads(post_data.decode('utf-8'))

            if 'review' in post_params:
                review_text = post_params['review']
                print(review_text)
                sentiment_score = predict_sentiment(review_text, model, vectorizer)
                response = {"sentiment_score": str(sentiment_score)}
                print(response)
            else:
                response = {"error": "Missing 'review' parameter in JSON data."}

        except json.JSONDecodeError:
            # Handle JSON decoding error
            response = {"error": "Invalid JSON data in the request body."}

        # Send JSON response
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))

if __name__ == "__main__":
    # Load the model and vectorizer
    vectorizer = joblib.load('vectorizer.joblib')
    model = joblib.load('model.joblib')

    # Start the server
    from http.server import HTTPServer

    server_address = ('', 8000)
    httpd = HTTPServer(server_address, SentimentAnalysisHandler)
    print('Starting server on port 8000...')
    httpd.serve_forever()
