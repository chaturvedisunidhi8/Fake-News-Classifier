from google.colab import files
uploaded = files.upload()  
 
!unzip tweets.csv.zip
 
import pandas as pd
df = pd.read_csv("tweets.csv")

df.head() 
  

# Display the shape of the dataframe
print(df.shape)

# Check  missing values

print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Show first 5 rows
df.head()

# Independent feature (text)
X = df['title'] 

# Dependent label
y = df['label']

print(X.shape)
print(y.shape)

import tensorflow as tf
print(tf.__version__)


from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

voc_size=5000

message=X.copy()
message['title'][1]
print(messages)

messages.reset_index(in place=True)
print (message)

import nltk
import re
from nltk.corpus import  stopwords

nltk.download ('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
corpus = []

for i in range(0, len(message)):

    review = re.sub('[^a-zA-Z]', ' ', message[i])
    review = review.lower()
    review = review.split()


    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#Onehot representation

onehot_repr=[one_hot(words, voc_size)for words in corpus]
print(onehot_repr)

corpus[1]
onehot_repr[1]

# Embedding representation
sent_length = 20

# Pad sequences
embedded_docs = pad_sequences(one_hot, padding='pre', maxlen=sent_length)

# Print all padded sequences
print(embedded_docs)

# Print the first padded

print( embedded_docs[0])


embedding_vector_features = 40
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_features, input_length=sent_length))
#model.add(LSTM(100))
model.add(Bidirectional (LSTM(100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
print(model.summary())
print(len(embedded_docs))
print(y.shape)

import numpy as np
x_final = np.array(embedded_docs)
y_final = np.array(y)

print(x_final.shape)
print(y_final.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33, random_state=42)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)
#Adding Dropout
from tensorflow.keras.layers.import Dropout

embedded_vector_features=40
Model=Sequential()
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


y_pred = model.predict(x_test)
y_pred = np.where(y_pred > 0.5, 1, 0)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
classification_report(y_test, y_pred)
