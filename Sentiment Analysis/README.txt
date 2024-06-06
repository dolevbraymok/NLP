the project is an implementation of Sentiment Analysis to decide if a tweet is positive or negative.
The model is ready to be tried in the file "sentiment_analysis_main.py"


the model is using dataset from NLTK library and in the "sentiment_analysis_model.py" the model is trained notice that if you want to use this file you might need to download the part below the imports

preprocessing is done using the regex of the form pattern_(*) and WordNetLemmatizer:
	i remove the hyperlinks, user names, and the signs for hashtag( not the whole hashtag as it might contain usefull information), and the sign for retweets

for feature extraction and model i did a few tries the best result were shown with logstic regression for model and with BoW of unigrams and bigrams for feature extraction