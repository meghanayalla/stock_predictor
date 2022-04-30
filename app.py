
import nltk
import re
nltk.download('stopwords')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras
import pickle
import numpy as np
from flask import Flask, render_template, request

model = keras.models.load_model("model_fin.h5")
with open('tokenizer_fin.pickle', 'rb') as handle:
   tokenizer = pickle.load(handle)
   

def preprocess(text):
    result = []
    stop_words = stopwords.words('english')

    # AFTER LOOKING AT THE DATA, IT WAS OBSERVED THAT SOME WORDS WERE FREQUENTLY REPEATED, THUS ADDING THEM IN THE STOPWORDS LIST (TO BE REMOVED FROM THE DATASET)
    stop_words.extend(['from', 'subject', 'https', 're', 'edu', 'use','will','aap','co','day','user','stock','today','week','year'])

    for token in gensim.utils.simple_preprocess(text):
        if len(token) >=3 and token not in stop_words:
            result.append(token)
    return result


app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        test = request.form['message']
        if len(test.split())<3:
            return render_template('result.html', result="Invalid Input")
        else:  
            test =preprocess(text=test)
            tokenized =tokenizer.texts_to_sequences([test])
            padded = pad_sequences(tokenized, maxlen=15, padding = 'post', truncating = 'post')
            pred = model.predict(padded)
            result = np.argmax(pred) 
            if result:
                return render_template('result.html', result="The Stock looks positive and promising")
            else:
                return render_template('result.html', result="Not the right time to invest.")

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
