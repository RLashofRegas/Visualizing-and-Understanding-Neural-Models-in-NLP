if [ -z ${1+x} ]; 
then 
    MODEL="model";
else 
    MODEL=$1;
fi

python util/StringToNum.py data/dict.txt input.txt util/input_index.txt
th saliency_derivative.lua $MODEL
python heatmap.py derivative-heatmap-$MODEL
