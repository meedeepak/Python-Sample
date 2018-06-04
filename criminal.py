# -*- coding: utf-8 -*-

import pandas as pd
dataset = pd.read_csv('criminal_train.csv')
testset = pd.read_csv('criminal_test.csv')

x_train=dataset.iloc[:,1:-1].values
y_train=dataset.iloc[:,-1].values
x_test=testset.iloc[:,1:].values

#x=pd.get_dummies(x_train,drop_first=True)
#test=pd.get_dummies(x_test,drop_first=True)
#test=test.reindex(columns = x.columns, fill_value=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x=sc.fit_transform(x_train)
test = sc.transform(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

neurons=x.shape[1]

#%%
model = Sequential()

model.add(Dense(units=neurons, kernel_initializer='uniform', activation='relu', input_dim=neurons))

model.add(Dense(units=int(neurons/2), kernel_initializer='uniform', activation='relu'))

model.add(Dense(units=int(neurons/6), kernel_initializer='uniform', activation='relu'))

model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x,y_train, batch_size = 32, epochs = 10)
#%%

y_pred = model.predict(test)

import numpy as np
y =  np.where(y>0.5,1,0)

sub=np.hstack((testset.iloc[:,0].values.reshape(y.shape),y))
sub=pd.DataFrame(sub)
sub.to_csv('submission.csv')
