if [ -z ${1+x} ]; 
then 
    MODEL="model";
else 
    MODEL=$1;
fi

if [ -z ${2+x} ]; 
then 
    DIM="60";
else 
    DIM=$2;
fi

python util/StringToNum.py data/dict.txt input.txt util/input_index.txt
th saliency_derivative.lua $DIM $MODEL
python heatmap.py derivative-heatmap-$MODEL
