#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 15:28:15 2018

@author: deepak
"""

import pandas as pd
import numpy as np
from nltk import word_tokenize, pos_tag
from keras.preprocessing.text import Tokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D

#list of words which are irrelevant for prediction
noise_words = ["...", "hmm", ":", ","]

#initialize required modules
stoplist = stopwords.words('english')
lem = WordNetLemmatizer()
stem = PorterStemmer()
t = Tokenizer()

train_set = pd.read_csv("train.csv");
test_set =  pd.read_csv("test.csv");

train = train_set.iloc[:, 3]
train_label = train_set.iloc[:, 4]
test = test_set.iloc[:, 4]

def preprocess(sentence):
    #split full sentence to array of words
    all_words = word_tokenize(sentence)
    
    #parts of speech tagging with each words keeps the context of a sentence ever if they are shuffled
    words_with_pos = pos_tag(all_words)
    
    #remove the stop words and lemmatize (getting root meaning of word)
    noise_free_words = [lem.lemmatize(word[0], "v")+"_"+word[1] for word in words_with_pos if (word[0] not in noise_words) and ( not word[0] in stoplist)]
    
    #needed join so that keras Tokenizer can work
    return " ".join(noise_free_words)


train_processed = [preprocess(text) for text in train]
test_processed = [preprocess(text) for text in test]
t.fit_on_texts(train_processed)
t.fit_on_texts(test_processed)

#create matrix of the each sentence
encoded_train = t.texts_to_matrix(train_processed, mode='count')
encoded_test = t.texts_to_matrix(test_processed, mode='count')

#define the model
#convolution model: A feature extraction model that learns to extract features from text
model = Sequential()
model.add(Embedding(encoded_train.shape[0] , 100, input_length = encoded_train.shape[1]))  
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())    
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#start training
model.fit(encoded_train, train_label, batch_size = 32, epochs = 10) 

#make prediction on test
test_pred = model.predict(encoded_test)
y =  np.where(test_pred>0.5,1,0)

#prepare the result to save to a file
submission = np.hstack((test_set.loc[:, ["COMMENT_ID"]].values, y))  
submission = pd.DataFrame(submission) 
submission.columns = ["COMMENT_ID", "CLASS"]

submission.to_csv('submission.csv')
