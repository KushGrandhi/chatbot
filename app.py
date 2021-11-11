import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
import random
import json
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import requests 
import time 

model = tf.keras.models.load_model('chatbot.h5')

data = pickle.load( open( "learning_chatbot.pkl", "rb" ) )
words = data['words']
classes = data['classes']
intents = json.loads(open('intents.json').read())
intents = intents["intents"]

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    pass

    return(np.array(bag))




def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    
    return return_list

def classify_flask(sentence):
    
    prob_threshold = 0.50
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>prob_threshold]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    try:
        answer_class = return_list[0]
        answer_class = answer_class["intent"]
        for i in intents:
            if i['tag'] == answer_class:
                ans = i['responses']
                num = len(ans)
                answer_number = random.randrange(0,num)
                answer = ans[answer_number]
        response = answer
    except:
        response = "Can you rephrase and try?"
    return response

def news():
    url = 'https://newsapi.org/v2/top-headlines?country=in&apiKey=090129b20ca84810a0dfdce46f31d06e'
    try: 
        response = requests.get(url) 
    except: 
        print('error')
    news = json.loads(response.text)
    i=0
    fnews = ''
    for new in news['articles']: 
        i+=1
        fnews+="#####    News   ###### "+str(i) + ': ' 
        fnews+= str(new['title']) 
        
        fnews+=' -> url:     '
    
        
    
        fnews+=str(new['url']) + "              "
        if i>2:
            break
    return fnews
     

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return 'Hello World! im Lenny, Please go to /bot route for accessing the chatbot'

@app.route("/bot", methods=['GET','POST'])
def classify(): 
    return render_template('home.html')
@app.route('/get')
def getmsg():
    userText = str(request.args.get('msg'))
    final = classify_flask(userText)
    if final == 'Getting news ...':
        final = news()
    return final

if __name__ == "__main__":
    app.run(debug=False)

