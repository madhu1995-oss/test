# -*- coding: utf-8 -*-
"""server.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fm0P7hA1huHWcRs_K6zU0E_hSYnCSpop
"""

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint

def load_dataset(filename):
  df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
  print(df.head())
  intent = df["Intent"]
  unique_intent = list(set(intent))
  sentences = list(df["Sentence"].astype(str))
  
  return (intent, unique_intent, sentences)

intent, unique_intent, sentences = load_dataset("indus_english.csv")

unique_intent

print(sentences[:5])

nltk.download("stopwords")
nltk.download("punkt")

#define stemmer
#stemmer = LancasterStemmer()

def cleaning(sentences):
  words = []
  for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    #stemming
    words.append([i.lower() for i in w])
    
  return words

cleaned_words = cleaning(sentences)
print(len(cleaned_words))
print(cleaned_words[:2])

def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token

def max_length(words):
  return(len(max(words, key = len)))

word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))

def encoding_doc(token, words):
  return(token.texts_to_sequences(words))

encoded_doc = encoding_doc(word_tokenizer, cleaned_words)

def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

padded_doc = padding_doc(encoded_doc, max_length)

padded_doc[:5]

print("Shape of padded docs = ",padded_doc.shape)

#tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')

output_tokenizer.word_index

encoded_output = encoding_doc(output_tokenizer, intent)

encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)

encoded_output.shape

def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))

output_one_hot = one_hot(encoded_output)

output_one_hot.shape

from sklearn.model_selection import train_test_split

train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)

print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))

def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(LSTM(128)))
  #model.add(Bidirectional(LSTM(128)))
  model.add(Dense(32, activation = "relu"))
  #model.add(Dropout(0.5))
  model.add(Dense(17, activation = "softmax"))
  
  return model

model = create_model(vocab_size, max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()

filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#hist = model.fit(train_X, train_Y, epochs = 10, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])

model1 = load_model("model.h5")

def predictions(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = word_tokenizer.texts_to_sequences(test_word)
  print(test_word)
  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))
    
  test_ls = np.array(test_ls).reshape(1, len(test_ls))
 
  x = padding_doc(test_ls, max_length)
  
  pred = model1.predict_proba(x)
  
  
  return pred





def get_final_output(pred, classes):
  predictions = pred[0]
 
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
 
  #for i in range(pred.shape[1]):
  print( (classes[0]))

import config
app = Flask(__name__) 
@app.route("/")
def predict():
  text = "Go to Language List"
  pred = predictions(text)
  return get_final_output(pred, unique_intent)

if __name__ == '__main__':
  app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)