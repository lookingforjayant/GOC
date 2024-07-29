import datetime
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
# from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
# import math

# nltk tools
import nltk
# nltk.download('stopwords') # needed to be downloaded one time
# nltk.download('wordnet') # needed to be downloaded one time
from nltk.corpus import stopwords

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


def lemmatize_text(review):
    word_as_tokens = w_tokenizer.tokenize(review)
    # print(word_as_tokens)
    return ' '.join([lemmatizer.lemmatize(word) for word in word_as_tokens])


def remove_stop_words(review, stop_words):
    return ' '.join([word for word in review.split() if word not in stop_words])


def clean_sentence(review):
    remove_list = ''
    # removing HTML tag
    replaced_string = re.sub(r'(<(\w+\s*)/>)|(<\w+>)', '', review)
    # removing URL string
    replaced_string = re.sub(r'https://.*|http://.*', ' ', replaced_string)
    # removing non-alphanumeric characters
    replaced_string = re.sub(r'[^\w' + remove_list + ']', ' ', replaced_string)
    # replacing double or n spaces with single space
    replaced_string = re.sub(r'(\s+)', ' ', replaced_string)
    return replaced_string


def sentiment_analyzer():
    # reading imdb data
    df = pd.read_csv('IMDB Dataset.csv')

    # taking only first 10 rows to understand the process
    # df[column : rows]
    # df.loc[row_label, column_label]
    # df.iloc[row_position, column_position]

    df['review'] = df['review'].apply(lambda x: clean_sentence(x))

    # removing stop words
    stop_words = set(stopwords.words('english'))
    df['review'] = df['review'].apply(lambda x: remove_stop_words(x, stop_words))

    # lemmatizing words in review
    df['review'] = df['review'].apply(lambda x: lemmatize_text(x))

    # checking stats of df
    total_words_length = 0.0
    for review in df['review']:
        word_list = review.split()
        total_words_length = total_words_length + len(word_list)
    print(f"Average length of each review : {total_words_length / df.shape[0]}")

    # percentage of review
    positive_review = 0
    for i in range(df.shape[0]):
        if df.iloc[i]['sentiment'] == 'positive':
            positive_review = positive_review + 1
    negative_review = df.shape[0] - positive_review
    print("Percentage of reviews with positive sentiment is " + str(positive_review / df.shape[0] * 100) + "%")
    print("Percentage of reviews with negative sentiment is " + str(negative_review / df.shape[0] * 100) + "%")

    # preparing data for model
    reviews = df['review'].values
    labels = df['sentiment'].values
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # creating train test split
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels,
                                                                                  test_size=0.2,
                                                                                  stratify=encoded_labels)

    # Hyperparameters of the model
    vocab_size = 3000  # choose based on statistics
    oov_tok = ''
    embedding_dim = 100
    max_length = 200  # choose based on statistics, for example 150 to 200
    padding_type = 'post'
    trunc_type = 'post'

    # tokenize sentences
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    # convert train dataset to sequence and pad sequences
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding='post', maxlen=max_length)

    # convert Test dataset to sequence and pad sequences
    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    test_padded = pad_sequences(test_sequences, padding='post', maxlen=max_length)

    # creating model
    # model initialization
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # model summary
    print(model.summary())

    # model training
    num_epochs = 5
    history = model.fit(train_padded, train_labels,
                        epochs=num_epochs, verbose=1,
                        validation_split=0.1)

    # model evaluation
    prediction = model.predict(test_padded)
    # Get labels based on probability 1 if p>= 0.5 else 0
    prediction_labels = []
    for i in prediction:
        if i >= 0.5:
            prediction_labels.append(1)
        else:
            prediction_labels.append(0)
    print(f"Accuracy of prediction on test set : {accuracy_score(test_labels, prediction_labels)}")

    # reviews on which we need to predict
    sentence = ["The movie was very touching and heart whelming",
                "I have never seen a terrible movie like this",
                "the movie plot is terrible but it had good acting"]

    # convert to a sequence
    sequences = tokenizer.texts_to_sequences(sentence)

    # pad the sequence
    padded = pad_sequences(sequences, padding='post', maxlen=max_length)

    # Get labels based on probability 1 if p>= 0.5 else 0
    prediction = model.predict(padded)
    prediction_labels = []
    for i in prediction:
        if i >= 0.5:
            prediction_labels.append(1)
        else:
            prediction_labels.append(0)

    for i in range(len(sentence)):
        print(sentence[i])
        if prediction_labels[i] == 1:
            s = 'Positive'
        else:
            s = 'Negative'
        print("Predicted sentiment : ", s)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print('Running as main script')
    print("=" * 50)
    sentiment_analyzer()
    print("=" * 50)
    print(f'Time taken by script | {datetime.datetime.now() - start}')
    print("=" * 50)
