from LRP.code.LSTM.LSTM_bidi import LSTM_bidi
from LRP.code.util.heatmap import html_heatmap

import codecs
import numpy as np
from sys import argv

model_path = argv[1]
model_hidden_dim = argv[2]
model_embedding_dim = argv[3]
output_file_name = 'LRP_Output.txt'

def predict(words):
    """Returns the classifier's predicted class"""
    net = LSTM_bidi(model_path) # load trained LSTM model
    w_indices = [net.voc.index(w) for w in words] # convert input sentence to word IDs
    net.set_input(w_indices) # set LSTM input sequence
    scores = net.forward() # classification prediction scores
    return np.argmax(scores)

def getWords():
    words_file = open('./tests_words.txt', 'r')
    words = words_file.readlines()
    words_file.close()
    return words

def getLRP(words, target_class):
    eps = 0.001
    bias_factor = 0.0
    net = LSTM_bidi(model_path) # load trained LSTM model
    w_indices = [net.voc.index(w) for w in words] # convert input sentence to word IDs
    Rx, Rx_rev, _ = net.lrp(w_indices, target_class, eps, bias_factor) # perform LRP
    R_words = np.sum(Rx + Rx_rev, axis=1) # compute word-level LRP relevances
    scores = net.s.copy() # classification prediction scores
    return scores, R_words

def runLRP():
    words = getWords()

    for sentence in words:
        sentence = sentence.replace('\n', '')
        word_list = sentence.split(' ')
        target_class = word_list[0]
        sentence_words = word_list[1:]
        predicted_class = predict(words)
        scores, R_words = getLRP(sentence_words, predicted_class)

        output_str = ',' + target_class + ',' + predicted_class + ',"' + str(sentence) + '","' + str(scores) + '","' + str(R_words) + '",' + html_heatmap(sentence, R_words)
        output_file = open(output_file_name, 'a')
        output_file.write(output_str)
        output_file.close()

runLRP()
