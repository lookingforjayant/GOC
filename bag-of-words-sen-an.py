# reading csv
import pandas as pd
# creating bag-of-words
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
# creating test-train split
from sklearn.model_selection import train_test_split
# NLP Model
from sklearn.naive_bayes import MultinomialNB
# testing model
from sklearn import metrics


def sentiment_analyzer():
    # reading csv
    df = pd.read_csv('training-data.csv')
    print(f'df shape {df.shape}')
    print(f'df columns {df.columns}')
    print("=" * 50)
    # creating bag-of-words
    print('Creating bag-of-words')
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    bag_of_words = cv.fit_transform(df['Sentence'])
    # creating test-train split
    x_train, x_test, y_train, y_test = train_test_split(bag_of_words, df['Sentiment'], test_size=0.2,
                                                        random_state=5)
    print("=" * 50)
    print("Training model")
    # training the model
    sen_an_model = MultinomialNB()
    sen_an_model.fit(x_train, y_train)
    print("=" * 50)
    print("Testing model")
    predicted = sen_an_model.predict(x_test)
    accuracy_score = metrics.accuracy_score(predicted, y_test)
    print(f"Accuracy | {accuracy_score}")
    print("=" * 50)


if __name__ == '__main__':
    print('Running as main script')
    sentiment_analyzer()
