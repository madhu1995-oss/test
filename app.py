#!/usr/bin/env python
# coding: utf-8

# In[53]:


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
from flask import Flask, request, jsonify


# In[8]:


def load_dataset(filename):
  df = pd.read_csv(filename, encoding = "latin1", names = ["Sentence", "Intent"])
  print(df.head())
  intent = df["Intent"]
  unique_intent = list(set(intent))
  sentences = list(df["Sentence"].astype(str))
  
  return (intent, unique_intent, sentences)
  


# In[9]:





# In[10]:


intent, unique_intent, sentences = load_dataset("indus_english.csv")


# In[11]:


unique_intent


# In[12]:


print(sentences[:5])


# In[13]:




# In[14]:


#define stemmer
#stemmer = LancasterStemmer()


# In[15]:


def cleaning(sentences):
  words = []
  for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    #stemming
    words.append([i.lower() for i in w])
    
  return words  


# In[16]:


cleaned_words = cleaning(sentences)
print(len(cleaned_words))
print(cleaned_words[:2])  
  


# In[17]:


def create_tokenizer(words, filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token


# In[18]:


def max_length(words):
  return(len(max(words, key = len)))
  


# In[19]:


word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))


# In[20]:


def encoding_doc(token, words):
  return(token.texts_to_sequences(words))


# In[21]:


encoded_doc = encoding_doc(word_tokenizer, cleaned_words)


# In[22]:


def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))


# In[23]:


padded_doc = padding_doc(encoded_doc, max_length)


# In[24]:


padded_doc[:5]


# In[25]:


print("Shape of padded docs = ",padded_doc.shape)


# In[26]:


#tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters = '!"#$%&()*+,-/:;<=>?@[\]^`{|}~')


# In[27]:


output_tokenizer.word_index


# In[28]:


encoded_output = encoding_doc(output_tokenizer, intent)


# In[29]:


encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)


# In[30]:


encoded_output.shape


# In[31]:


def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))


# In[32]:


output_one_hot = one_hot(encoded_output)


# In[33]:


output_one_hot.shape


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.2)


# In[36]:


print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))


# In[37]:


def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(LSTM(128)))
  #model.add(Bidirectional(LSTM(128)))
  model.add(Dense(32, activation = "relu"))
  #model.add(Dropout(0.5))
  model.add(Dense(17, activation = "softmax"))
  
  return model


# In[38]:


model = create_model(vocab_size, max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()


# In[39]:


filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#hist = model.fit(train_X, train_Y, epochs = 10, batch_size = 32, validation_data = (val_X, val_Y), callbacks = [checkpoint])


# In[40]:


model1 = load_model("model.h5")


# In[41]:


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


  


# In[ ]:





# In[ ]:





# In[42]:


def get_final_output(pred, classes):
  predictions = pred[0]
 
  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)
 
  #for i in range(pred.shape[1]):
  return  classes[0]


# In[43]:

app = Flask(__name__)
@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    pred = predictions(data["text"])
    a=get_final_output(pred, unique_intent)
    return jsonify(a)
if __name__ == '__main__':
    app.run(port=9000, debug=True)

# In[ ]:




