import tensorflow as tf
from tensorflow import keras
import json
import pickle
import random
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
# from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words=[]
classes=[]
documents=[]
ignore_letters=['?','!',',','.']
for intent in intents['intents']:
  for pattern in intent['patterns']:
    wordlist=nltk.word_tokenize(pattern)
    words.extend(wordlist)
    documents.append((wordlist,intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])


words =[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))
print(words)
print(classes)

training=[]
output_empty =[0]*len(classes)

for document in documents:
  bag=[]
  word_patterns=document[0]
  word_patterns=[lemmatizer.lemmatize(word.lower())for word in word_patterns]
  for word in words:
    bag.append(1) if word in word_patterns else bag.append(0)
  output_row= list(output_empty)
  output_row[classes.index(document[1])]=1
  training.append([bag, output_row])
print(training)
print(bag)

random.shuffle(training)
training=np.array(training)

train_X=list(training[:,0])
train_Y=list(training[:,1])

model= Sequential()
model.add(Dense(128, input_shape=(len(train_X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_Y[0]),activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
hist=model.fit(np.array(train_X), np.array(train_Y), epochs=200, batch_size=5, verbose=1)
model.save('CodeClause_Chatbot.h5',hist)
print("Done")