import pickle
from sentiment_analysis_model import preprocess_tweets, patterns
from nltk.stem import WordNetLemmatizer
import numpy as np




if __name__ == '__main__':
    # Load the model
    with open('logistic_regression_model.pkl', 'rb') as file:
        logistic_regression_model = pickle.load(file)
    with open('count_vectorizer.pkl', 'rb') as file:
        count_vectorizer = pickle.load(file)

    lemmatizer = WordNetLemmatizer()
    while True:
        print("Insert Tweet or quit: ")
        user_input = input().strip()
        if user_input=="quit":
            break
        tweet = np.array([lemmatizer.lemmatize(*preprocess_tweets([user_input], patterns))])
        tweet = count_vectorizer.transform(tweet)
        prediction = logistic_regression_model.predict(tweet)
        if prediction[-1] == 0:
            out = "negative"
        else:
            out ="positive"
        print(f"the statment:{user_input} is {out}")
    print("GoodBye")