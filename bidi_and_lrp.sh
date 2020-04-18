for i in 2 3 # 2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 20
do
    for j in 2 # 2 3 4 5 6 7 8 9 10 11 12 13 14 15 17 20
    do
        MODEL_NAME="model_dim${i}_emb${j}"
        th ./sentiment_bidi/main.lua $i $j $MODEL_NAME
        python run_lrp.py /home/rlashof/repos/Visualizing-and-Understanding-Neural-Models-in-NLP/sentiment_bidi/${MODEL_NAME}-00
    done
done