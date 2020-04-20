from LRP.code.LSTM.LSTM_bidi import LSTM_bidi
from LRP.code.util.heatmap import html_heatmap

import codecs
import numpy as np

def predict(words, model_path):
    """Returns the classifier's predicted class"""
    net = LSTM_bidi(model_path) # load trained LSTM model
    w_indices = [net.voc.index(w) for w in words] # convert input sentence to word IDs
    net.set_input(w_indices) # set LSTM input sequence
    scores = net.forward() # classification prediction scores
    return np.argmax(scores)

def getLRP(words, target_class, model_path):
    eps = 0.001
    bias_factor = 0.0
    net = LSTM_bidi(model_path) # load trained LSTM model
    w_indices = [net.voc.index(w) for w in words] # convert input sentence to word IDs
    Rx, Rx_rev, _ = net.lrp(w_indices, target_class, eps, bias_factor) # perform LRP
    R_words = np.sum(Rx + Rx_rev, axis=1) # compute word-level LRP relevances
    scores = net.s.copy() # classification prediction scores
    return scores, R_words

def getLRPForArray(words, model_path, target_class = None):
    predicted_class = predict(words, model_path)
    if target_class == None:
        target_class = predicted_class
    scores, R_words = getLRP(words, target_class, model_path)
    output_dict = {}
    output_dict['predicted_class'] = predicted_class
    output_dict['scores'] = scores.tolist()
    output_dict['relevances'] = R_words.tolist()
    output_dict['html_heatmap'] = html_heatmap(words, R_words)
    return output_dict