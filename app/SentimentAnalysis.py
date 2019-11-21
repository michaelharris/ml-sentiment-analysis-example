import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
# Define a class
class TextProcessor:
    def __init__(self):
        # we're still going to use a Tokenizer here, but we don't need to fit it
        self.tokenizer = Tokenizer(num_words=3000)
        # for human-friendly printing
        self.labels = ['negative', 'positive']

        # read in our saved dictionary
        with open('../models/dictionary.json', 'r') as dictionary_file:
            self.dictionary = json.load(dictionary_file)
        
        json_file = open('../models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        # and create a model from that
        self.model = model_from_json(loaded_model_json)
        # and weight your nodes with your saved values
        self.model.load_weights('../models/model.h5')

        # this utility makes sure that all the words in your input
    # are registered in the dictionary
    # before trying to turn them into a matrix.
    def convert_text_to_index_array(self, text):
        words = kpt.text_to_word_sequence(text)
        wordIndices = []
        for word in words:
            if word in self.dictionary:
                wordIndices.append(self.dictionary[word])
            #else:
                #print("'%s' not in training corpus; ignoring." %(word))
        return wordIndices


    def evaluate_text(self, evalSentence):
        # format your input for the neural net
        testArr = self.convert_text_to_index_array(evalSentence)
        preparedInput = self.tokenizer.sequences_to_matrix([testArr], mode='binary')
        # predict which bucket your input belongs in
        pred = self.model.predict(preparedInput)
        print("Positivity rating; %f%%" %  (pred[0][0] * 100))
        
        return pred[0][0]


    