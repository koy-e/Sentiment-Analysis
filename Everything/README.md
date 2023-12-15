# Group 1

## Instructions to Train The Model

### Multi-class Classification

- Within the MultiClassExperiment folder, you will find the code we used to do our initial multi class classification experiment. It will output all important data of each of the model's training into a `.txt` file.

### Binary Classification

- Within BinaryClassificationExperiment folder, you will find the three models we used to train for predicting sentiment from Amazon reviews, in their own respective folder.
- Each model, once run, will output a `.txt` file containing all important information about the model's training.

### Realtime Sentiment Feedback

- Within the RealTimeSentimentFeedbackProgram folder, you will find the code `RealTimeSentimentFeedback.py` we used to train and save the best parameter Support Vector Machine model along with TfidfVectorizer. Both model and vectorizer are saved as a `.joblib` file and are used in `CreateModelandVectorizer.py` to provide realtime sentiment feedback to user input reviews.
- `CreateModelandVectorizer.py` is a currently written to only handle command line inputs and provide sentiment outputs.

```
In order to run all codes, please ensure you have installed all the required libraries and have python on your computer.
```

Original dataset is located at: [Kaggle](https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset?select=amazon_reviews_us_Apparel_v1_00.tsv)

We have included our training dataset `DSB.csv` in the folder.

All Tables and Figures used in the report that showcase our findings can be found in the file `TablesAndFigures.xlsx`
