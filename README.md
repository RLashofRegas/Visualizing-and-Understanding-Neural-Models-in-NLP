# Visualizing and Understanding Neural Models in NLP
This is a fork from https://github.com/jiweil/Visualizing-and-Understanding-Neural-Models-in-NLP for implementation of my final project. Additional code has been utilized from https://github.com/ArrasL/LRP_for_LSTM for implementing layer-wise relevance propagation.

## Requirements:
GPU

Torch (nn,cutorch,cunn,nngraph)

python [matplotlib library](http://matplotlib.org/users/installing.html) (only for matrix plotting purposes)

download [data](http://cs.stanford.edu/~bdlijiwei/visual_data.tar)

## Run the models:

sh bidi_and_lrp.sh 

This command creates trained models with a large array of hidden and embedding dimensions and takes about 12 hours to run on my NVIDIA GTX 980 GPU.

In order to run the analysis notebook rename the file from LRP_Output.txt to LRP_Output_Large.csv and add the following header:
hidden_dimensions|embedding_dimensions|fine_accuracy|coarse_accuracy|true_class|predicted_class|sentence|class_scores|word_relevances|heatmap_html

The analysis can then be ran using LRP_Analysis_Large.ipynb
