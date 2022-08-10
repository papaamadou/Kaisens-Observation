import nltk
from nltk.corpus import stopwords
import re  # regular expression
import os
import joblib
import uvicorn
from fastapi import FastAPI 

app = FastAPI(
    title="Toxicity of Twitter",
    description="A simple API that use NLP model to predict the toxicity of a tweet",
    version="0.1",
)

with open("models/toxicity_classifier_pipeline.pkl", "rb" ) as f:
    model = joblib.load(f)

def text_cleaning(text, remove_stop_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = " ".join(re.sub("(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", text).split())
    #On remplace les retours de chariots par des espaces
    text = text.replace('\\n', ' ')
   #On enlev les emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
     #On enleve les pseudo des utilisateurs dans les tweets et on met le text en minuscule
    tokens = text.split()
    tokens = [i.lower() for i in tokens if '@' not in i]
    text =  " ".join(tokens)
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("french")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Return a list of words
    return text

@app.get("/predict-tweet")
def predict_sentiment(tweet: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_tweet = text_cleaning(tweet)
    
    # perform prediction
    prediction = model.predict([cleaned_tweet])
    output = int(prediction[0])
    #probas = model.predict_proba([cleaned_tweet])
    #output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}
    
    # show results
    #result = {"prediction": sentiments[output], "Probability": output_probability}
    result = {"prediction": sentiments[output]}
    return result

