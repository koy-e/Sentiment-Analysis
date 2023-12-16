import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import re
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
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


if __name__ == "__main__":
    main()
