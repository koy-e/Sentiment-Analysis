# import pandas as pd
# import nltk
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import MultinomialNB
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from nltk.tokenize import word_tokenize
# nltk.download('all')
# import numpy as np

# import sys

# # Load data
# df = pd.read_csv('db.tsv')

# df.head(4)



# # Clean data
# df = df.dropna()

# print(df.i)
# # Text normalization and tokenization
# df['review_body'] = df['review_body'].str.lower()
# df['review_body'] = df['review_body'].apply(word_tokenize)
# df['review_body'] = df['review_body'].apply(lambda x: [word.translate(str.maketrans('', '', string.punctuation)) for word in x])
# df['review_body'] = df['review_body'].apply(lambda x: [word for word in x if word])

# # Stop words removal
# stop_words = set(stopwords.words('english'))
# df['review_body'] = df['review_body'].apply(lambda x: [word for word in x if word not in stop_words])

# # Stemming
# porter = PorterStemmer()
# df['review_body'] = df['review_body'].apply(lambda x: [porter.stem(word) for word in x])

# df['review_body'].head(5)


# # Join the words back into a single string
# df['review_body'] = df['review_body'].apply(lambda x: ' '.join(x))

# df['review_body'].head(5)


# # Vectorization
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df['review_body'])

# # Now X can be used to train a machine learning model


# # Vectorization
# vectorizers = [
#     ("CountVectorizer", CountVectorizer()),
#     ("TfidfVectorizer", TfidfVectorizer()),
#     ("HashingVectorizer", HashingVectorizer())
# ]

# for name, vectorizer in vectorizers:
#     X = vectorizer.fit_transform(df['review_body'])

#     # Splitting the data
#     X_train, X_test, y_train, y_test = train_test_split(X, df['star_rating'], test_size=0.2, random_state=42)

#     # Training the model
#     model = LogisticRegression()
#     model.fit(X_train, y_train)

#     # Evaluating the model
#     score = model.score(X_test, y_test)
#     print(f"{name} score: {score}")


#     # Evaluating the model
# y_pred = model.predict(X_test)
# print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
# print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
# print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")
# print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")

# # Tuning the model's hyperparameters
# param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
# grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
# grid.fit(X_train, y_train)
# print(f"Best parameters: {grid.best_params_}")

# # Trying different machine learning models
# models = [
#     ("RandomForestClassifier", RandomForestClassifier()),
#     ("SVC", SVC()),
#     ("MultinomialNB", MultinomialNB())
# ]

# for name, model in models:
#     model.fit(X_train, y_train)
#     score = model.score(X_test, y_test)
#     print(f"{name} score: {score}")

#     # Interpreting the model's results
# feature_names = vectorizer.get_feature_names_out()
# coefficients = model.coef_[0]
# indices = np.argsort(coefficients)

# # Most positive words
# print("Most positive words:")
# for i in range(10):
#     print(feature_names[indices[-i-1]])

# # Most negative words
# print("\nMost negative words:")
# for i in range(10):
#     print(feature_names[indices[i]])

# # Improving the model based on the evaluation
# # If you want to improve recall (minimize false negatives)
# threshold = 0.4  # Adjust this value based on your needs
# y_pred = (model.predict_proba(X_test)[:,1] >= threshold).astype(bool)
# print(f"New Recall: {recall_score(y_test, y_pred)}")

# # If you want to improve precision (minimize false positives)
# threshold = 0.6  # Adjust this value based on your needs
# y_pred = (model.predict_proba(X_test)[:,1] >= threshold).astype(bool)
# print(f"New Precision: {precision_score(y_test, y_pred)}")








import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np

# Download NLTK resources
nltk.download('all')

# Load data
df = pd.read_csv('db.tsv', sep='\t')

# Clean data
df = df.dropna()

# Text normalization and tokenization
df['review_body'] = df['review_body'].str.lower()
df['review_body'] = df['review_body'].apply(word_tokenize)
df['review_body'] = df['review_body'].apply(lambda x: [word.translate(str.maketrans('', '', string.punctuation)) for word in x])
df['review_body'] = df['review_body'].apply(lambda x: [word for word in x if word])

# Stop words removal
stop_words = set(stopwords.words('english'))
df['review_body'] = df['review_body'].apply(lambda x: [word for word in x if word not in stop_words])

# Stemming
porter = PorterStemmer()
df['review_body'] = df['review_body'].apply(lambda x: [porter.stem(word) for word in x])

# Join the words back into a single string
df['review_body'] = df['review_body'].apply(lambda x: ' '.join(x))

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review_body'])

# Vectorization
vectorizers = [
    ("CountVectorizer", CountVectorizer()),
    ("TfidfVectorizer", TfidfVectorizer()),
    ("HashingVectorizer", HashingVectorizer())
]

for name, vectorizer in vectorizers:
    X = vectorizer.fit_transform(df['review_body'])

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, df['star_rating'], test_size=0.2, random_state=42)

    # Training the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluating the model
    score = model.score(X_test, y_test)
    print(f"{name} score: {score}")

    # Evaluating the model
y_pred = model.predict(X_test)
print(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
print(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")
print(f"Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}")

# Tuning the model's hyperparameters
param_grid = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best parameters: {grid.best_params_}")

# Trying different machine learning models
models = [
    ("RandomForestClassifier", RandomForestClassifier()),
    ("SVC", SVC()),
    ("MultinomialNB", MultinomialNB())
]

for name, model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} score: {score}")

    # Interpreting the model's results
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
indices = np.argsort(coefficients)

# Most positive words
print("Most positive words:")
for i in range(10):
    print(feature_names[indices[-i-1]])

# Most negative words
print("\nMost negative words:")
for i in range(10):
    print(feature_names[indices[i]])

# Improving the model based on the evaluation
# If you want to improve recall (minimize false negatives)
threshold = 0.4  # Adjust this value based on your needs
y_pred = (model.predict_proba(X_test)[:,1] >= threshold).astype(bool)
print(f"New Recall: {recall_score(y_test, y_pred)}")

# If you want to improve precision (minimize false positives)
threshold = 0.6  # Adjust this value based on your needs
y_pred = (model.predict_proba(X_test)[:,1] >= threshold).astype(bool)
print(f"New Precision: {precision_score(y_test, y_pred)}")
