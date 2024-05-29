import nltk
import pickle
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import re
"""# Download NLTK resources
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')"""

def preprocess_tweets(tweets, patterns):
    preprocessed = np.empty(len(tweets),dtype=object)
    for i, tweet in enumerate(tweets):
        processed_text = tweet
        for pattern in patterns:
            processed_text = re.sub(pattern, '', processed_text)

        preprocessed[i] = processed_text
    return preprocessed



# Load positive, negative, and neutral tweets from NLTK's twitter_samples corpus
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
neutral_tweets = twitter_samples.strings('tweets.20150430-223406.json')



#Pattern for hyperlinks @USER and #hashtags and RT of retweets
pattern_hyperlink = r'https?:\/\/.*'
pattern_user = r'@[\w]*'
pattern_hashtag = r'#'  # removing only the hashtag sign itself as the tag may contain information
pattern_retweet = r'^RT[\s]+'

# Notice that i dont remoce punctuations as something like ":)" can be important for the specific dataset

#preprocessing of patterns
patterns = [pattern_hyperlink, pattern_user, pattern_hashtag, pattern_retweet]

positive_tweets = preprocess_tweets(positive_tweets, patterns)
negative_tweets = preprocess_tweets(negative_tweets, patterns)
neutral_tweets = preprocess_tweets(neutral_tweets, patterns)

tweets = np.concatenate((positive_tweets, negative_tweets, neutral_tweets))
labels = [1]*len(positive_tweets) + [0]*len(negative_tweets) + [2]*len(neutral_tweets) # creating fitting labels for each tweet


#tokeniaztion and lemmaniza
lemmatizer = WordNetLemmatizer()
tweets = [lemmatizer.lemmatize(tweet) for tweet in tweets]

#bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow_vectorizer = CountVectorizer(ngram_range=(1,2))
bow = bow_vectorizer.fit_transform(tweets)

X_train, X_test, y_train, y_test = train_test_split(bow, labels, test_size=0.2, random_state=51)

print("****************************LOGISTIC REGRESSION*********************************")
logistic_regression = LogisticRegression(max_iter=1000)  # Initialize logistic regression classifier
logistic_regression.fit(X_train, y_train)  # Train the classifier
print("****************************FINISHED TRAINING*********************************")

# Predict labels for test data
y_pred = logistic_regression.predict(X_test)

# Generate classification report
report = classification_report(y_test, y_pred)

print("Classification Report:\n", report)

"""print("****************************RANDOM FOREST*********************************")

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # Initialize Random Forest classifier with 100 trees
random_forest.fit(X_train, y_train)  # Train the classifier
print("****************************FINISHED TRAINING*********************************")

# Predict labels for test data
y_pred = random_forest.predict(X_test)
# Generate classification report
report = classification_report(y_test, y_pred)

print("Classification Report:\n", report)"""



with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(logistic_regression, f)

"""# Save Random Forest model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(random_forest, f)
"""
# Save CountVectorizer
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(bow_vectorizer, f)

print("saved")