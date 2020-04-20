from LRP.code.LSTM.LSTM_bidi import LSTM_bidi
from LRP.code.util.heatmap import html_heatmap

import codecs
import numpy as np
from sys import argv

model_path = argv[1]
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

    output_file = open(output_file_name, 'r')
    existing_lines = output_file.readlines()
    last_line = existing_lines[-1]
    output_file.close()

    output_file = open(output_file_name, 'a')
    is_first = True
    for sentence in words:
        sentence = sentence.replace('\n', '').strip()
        word_list = sentence.split(' ')
        target_class = word_list[0] - 1 # lua model goes off of 1-indexed classes
        sentence_words = word_list[1:]
        predicted_class = predict(sentence_words)
        scores, R_words = getLRP(sentence_words, predicted_class)

        output_str = '|' + str(target_class) + '|' + str(predicted_class) + '|' + str(sentence_words) + '|' + str(scores.tolist()) + '|' + str(R_words.tolist()) + '|' + html_heatmap(sentence_words, R_words)
        if not is_first:
            output_str = '\n' + last_line + output_str
        else:
            is_first = False
        output_file.write(output_str)

    output_file.close()

runLRP()
