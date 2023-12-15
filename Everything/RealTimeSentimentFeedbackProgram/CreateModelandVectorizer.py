# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import string
import time
import re
import joblib  # To save the vectorizer and model

# Make sure to download stopwords from nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # Added for lemmatization

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

    # Handling negations
    words = [words[i] + '_' + words[i + 1] if words[i] == 'not' and i +
             1 < len(words) else words[i] for i in range(len(words))]

    lemmatized = [lemmatizer.lemmatize(
        word) for word in words if word.isalpha() and word not in stop_words]

    return ' '.join(lemmatized)


# Load data
df = pd.read_csv('DSB.csv')

# Filter out the 3-star reviews
df = df[df['star_rating'] != 3]

# Remove missing values
df = df.dropna(subset=['review_body', 'star_rating'])

# Remove HTML tags from review text
df['review_body'] = df['review_body'].apply(remove_html_tags)

# Preprocess text data
df['review_body'] = df['review_body'].apply(preprocess_text)

# Categorizing Ratings into 'positive' or 'negative'


def categorize_rating(rating):
    if rating >= 4:
        return 'positive'
    else:  # rating of 1 or 2
        return 'negative'


df['sentiment'] = df['star_rating'].apply(categorize_rating)

# Define vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 3))

# Vectorizing the text
X = vectorizer.fit_transform(df['review_body'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the model with probability estimation
model = SVC(kernel='linear', C=1, probability=True)
model.fit(X_train, y_train)

# Predict probabilities
probabilities = model.predict_proba(X_test)

# Normalize probabilities to be between -1 and 1
# Assuming positive class is the second column
normalized_scores = 2 * probabilities[:, 1] - 1

# Model Evaluation
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, model.predict(X_test)))

# Save the vectorizer and model
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'model.joblib')
