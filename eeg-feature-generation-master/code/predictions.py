import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from tensorflow.python.keras.layers import Dense, Activation, Flatten, InputLayer, Dropout, LSTM
from sklearn.preprocessing import LabelEncoder
import EEG_generate_training_matrix
import os

def split_dataframe(df, chunk_size):
    return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

model = tf.keras.Sequential([
    InputLayer(input_shape=(988, 1)),
    tf.keras.layers.GRU(256, return_sequences=True),
    Flatten(),
    Dense(4, activation='relu'),
    Dropout(0.4),
    Dense(3, activation="softmax")
])
model.load_weights('eeg-feature-generation-master\code\EEGmodel.weights.h5')

def find_emotion(chunk):
    answer = model.predict(chunk)
    print(answer)
    p = 0
    n = 0
    ne = 0
    for row in answer:
        if(row[0] > row[1] and row[0] > row[2]):
            n+=1
        elif(row[1] > row[2] and row[1] > row[0]):
            ne+=1
        else:
            p+=1
    if p>n and p>ne:
        return 'positive'
    elif n > p and n > ne:
        return 'negative'
    else:
        return 'neutral'

def predict_emotion(cleanedEEGcsv):
    emotions = []
    PATH = 'eeg-feature-generation-master\dataset\MUSE2/features.csv'
    EEG_generate_training_matrix.gen_training_matrix(cleanedEEGcsv, PATH, -1)
    data = pd.read_csv(PATH)
    chunks = split_dataframe(data, 5)
    for chunk in chunks:
        chunk = np.array(chunk).reshape((chunk.shape[0],chunk.shape[1],1))
        emotion = find_emotion(chunk)
        emotions.append(emotion)
    print(emotions)
    os.remove(PATH)
    return emotions
