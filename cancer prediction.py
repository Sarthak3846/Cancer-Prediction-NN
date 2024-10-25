import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\\Sarthak Tyagi\\Downloads\\Cancer_Data.csv")
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 'M' else 0) 
df.head()

df.drop(['id','Unnamed: 32'], axis='columns',inplace=True)

x = df.drop('diagnosis', axis='columns')
y = df['diagnosis']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential 
model = Sequential()

#add layers
model.add(tf.keras.layers.Dense(128, activation='relu',input_shape=(x_train.shape[1],)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
         

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train,epochs=20,batch_size=32, validation_split=0.2)