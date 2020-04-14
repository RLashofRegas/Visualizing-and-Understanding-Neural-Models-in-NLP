require "torchx"
local stringx = require('pl.stringx')
local function fromString(str,splitter)
    -- split string on splitter chars
    local split=stringx.split(str,splitter);
    local Return={};
    -- tensors of 0 of number of elements - 1 (first element is class label)
    local tensor=torch.Tensor(1,#split-1):zero()
    local tensor_r=torch.Tensor(1,#split-1):zero()
    for i=2,#split do
        -- word indices
        tensor[1][i-1]=tonumber(split[i]);
        -- word indices in reverse
        tensor_r[1][#split-i+1]=tonumber(split[i]);
    end
    -- return {class label, phrase, reverse phrase}
    return {tonumber(split[1]),tensor,tensor_r};
end


local function split(str)
    local split=stringx.split(str," ");
    local tensor=torch.Tensor(1,#split):zero()
    for i=1,#split do
        tensor[1][i]=tonumber(split[i]);
    end
    return tensor;
end

local function read_embedding()
    local timer=torch.Timer()
    time1=timer:time().real;
    local index=0;
    local V=19538;
    local D=300;
    tensor=torch.Tensor(V,D):zero();
    local index=0
    for line in io.lines("sentiment_glove_300.txt") do
        index=index+1;
        tensor:indexCopy(1,torch.LongTensor({index}),split(line));
    end
    local file=torch.DiskFile("embedding","w"):binary();
    file:writeObject(tensor);
    file:close()
    return tensor;
end


local function get_batch(texts,texts_r)
    -- get max length of phrases
    local max_length=-100;
    for i=1,#texts do   
        if texts[i]:size(2)>max_length then
            max_length=texts[i]:size(2)
        end
    end

    local Words=torch.Tensor(#texts,max_length):fill(1);
    local Words_r=torch.Tensor(#texts,max_length):fill(1);
    local mask=torch.Tensor(#texts,max_length):zero();
    for i=1,#texts do
        -- get row for this phrase, and only the last N columns where N is number of words
        Words:sub(i,i,max_length-texts[i]:size(2)+1,max_length):copy(texts[i]);
        Words_r:sub(i,i,max_length-texts[i]:size(2)+1,max_length):copy(texts_r[i]);
        mask:sub(i,i,max_length-texts[i]:size(2)+1,max_length):copy(texts[i]);
    end
    -- fill Mask - one row for each column containing the indexes of all zeros
    local Mask={};
    for i=1,Words:size(2) do
        Mask[i]=torch.LongTensor(torch.find(mask:sub(1,-1,i,i),0))
    end
    return Words,Words_r,Mask
end

local function read_train(open_train_file,batch_size)
    Y={}; texts={};texts_r={};
    i=0;
    End=0;
    while 1==1 do
        i=i+1;
        -- read next line from io (needs to be set to train file prior to calling this function)
        local str=io.read();
        -- break at the end of the file
        if str==nil then
            End=1
            break;
        end
        -- get {class label, phrase indices, reverse phrase indices}
        -- phrases are 1 row tensors with #columns = number of words
        local Split=fromString(str);
        -- class labels
        Y[i]=Split[1];
        -- forwards phrases
        texts[i]=Split[2];
        -- reverse phrases
        texts_r[i]=Split[3];
        -- only read one batch
        if i==batch_size then
            break;
        end
    end
    -- Words/Words_r - tensor of width max_length with word indices at the end of the row
    -- Mask - one row for each column of Words/Words_r with indices of all zeros.
    local Words,Words_r,Mask=get_batch(texts,texts_r);
    -- return whether we finished the file, tensor of class labels, tensors of word indices (forwards and backwards), indices of zeros
    return End,torch.Tensor(Y),Words,Words_r,Mask
end

return {read_train=read_train,read_embedding=read_embedding}

