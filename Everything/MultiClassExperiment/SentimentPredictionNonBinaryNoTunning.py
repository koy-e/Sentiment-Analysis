# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import string
import time  # Import the time module

# Make sure to download stopwords from nltk
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
stop_words = set(stopwords.words('english'))
htmlbr = ["br"]


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
        word) for word in words if word.isalpha() and word not in stop_words and word not in htmlbr]

    return ' '.join(lemmatized)


# Load data
df = pd.read_csv('DSB.csv')

# Preprocess text data
df['review_body'] = df['review_body'].apply(preprocess_text)

# Categorizing Ratings


def categorize_rating(rating):
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:  # rating of 1 or 2
        return 'negative'


df['sentiment'] = df['star_rating'].apply(categorize_rating)

# Feature Extraction
vectorizers = {
    "TfidfVectorizer": TfidfVectorizer(),
    "CountVectorizer": CountVectorizer()
}

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial NB": MultinomialNB(),
    "SVC": SVC(),
}

with open("NonBinaryNoTunning.txt", "w") as file:
    # Loop over vectorizers and models
    for vec_name, vectorizer in vectorizers.items():
        file.write(f"Using vectorizer: {vec_name}\n")
        print(f"Using vectorizer: {vec_name}")
        X = vectorizer.fit_transform(df['review_body'])
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        class_distribution = y_train.value_counts(normalize=True)
        file.write(
            f"Class Distribution in Training Data:\n{class_distribution}\n")
        print("Class Distribution in Training Data:\n", class_distribution)

        for model_name, model in models.items():
            start_time = time.time()  # Get the start time
            file.write(f"Training model: {model_name}\n")
            print(f"Training model: {model_name}")

            # Train the model
            model.fit(X_train, y_train)

            # Model Evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            file.write(
                f"Accuracy for {model_name} using {vec_name}: {accuracy}\n")
            file.write(classification_report(y_test, y_pred))
            print(f"Accuracy for {model_name} using {vec_name}: {accuracy}")
            print(classification_report(y_test, y_pred))

            end_time = time.time()  # Get the end time
            loop_duration = end_time - start_time  # Calculate the loop duration
            minutes = int(loop_duration // 60)  # Calculate the minutes
            seconds = int(loop_duration % 60)  # Calculate the seconds
            # Format the duration as hours:minutes:seconds
            hours = minutes // 60  # Calculate the hours
            minutes = minutes % 60  # Calculate the remaining minutes
            duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            file.write(
                f"Model {model_name} fit-train-predict duration: {duration_formatted}\n")
            print(f"time taken for model {model_name} is {duration_formatted}")
