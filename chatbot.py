import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import json
intents = json.loads(open(r'C:\Users\HP\PycharmProjects\CodeClause_Chatbot\intents.json').read())
lemmatizer = WordNetLemmatizer()
import nltk
import random

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('CodeClause_Chatbot.h5')


def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bags_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bags_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, result] for i, result in enumerate(result) if result > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for result in results:
        return_list.append({'intent': classes[result[0]], 'probability': str(result[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            returning = random.choice(i['responses'])
    return returning


print("do it ")

while True:
    message = input("")
    ints = predict_class(message)
    res = (get_response(ints, intents))
    print(res)
