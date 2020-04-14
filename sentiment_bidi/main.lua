require "cutorch"
require 'cunn'
require "nngraph"

local model={};
paramx={}
paramdx={}
ada={}
local LookupTable=nn.LookupTable;

local params={batch_size=1000,
    max_iter=20,
    dimension=60,
    dropout=0.2,
    train_file="../data/sequence_train.txt",
    --train_file="small",
    init_weight=0.1,
    learning_rate=0.05,
    dev_file="../data/sequence_dev_root.txt",
    test_file="../data/sequence_test_root.txt",
    max_length=100,
    vocab_size=19538
}

function g_cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end



local function lstm(x,prev_h,prev_c)
    -- x = each row is a word from a phrase embedded in "dimension" dimensions

    -- dropout weights from the input
    local drop_x=nn.Dropout(params.dropout)(x)
    -- dropout weights from the output of the previous layer
    local drop_h=nn.Dropout(params.dropout)(prev_h)
    -- dim x 4*dim linear layers on x and h (concatenate to create input gate)
    local i2h=nn.Linear(params.dimension,4*params.dimension)(drop_x);
    local h2h=nn.Linear(params.dimension,4*params.dimension)(drop_h);
    -- add tensor ouputs of i2h and h2h = n_phrases x 4*dim dimensional tensor
    local gates=nn.CAddTable()({i2h,h2h});
    -- reshape each row (current word from each phrase) into 4xdimension Tensor = n_phrasesx4xdimension
    local reshaped_gates =  nn.Reshape(4,params.dimension)(gates);
    -- split each of the 4 gates into separate tensors, 4 tensors of dim = n_phrasesxdimension
    local sliced_gates = nn.SplitTable(2)(reshaped_gates);
    -- get each of the 4 tensors as the gates
    local in_gate= nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform= nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate= nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate= nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))
    -- put gates together with incoming data
    local l1=nn.CMulTable()({forget_gate, prev_c})
    local l2=nn.CMulTable()({in_gate, in_transform})
    local next_c=nn.CAddTable()({l1,l2});
    local next_h= nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    return next_h,next_c
end

local function encoder_()
    -- x_index is passed in as one word for each
    local x_index=nn.Identity()();
    local prev_c=nn.Identity()();
    local prev_h=nn.Identity()();
    -- 60 dimensional embedding vectors, one for each word in vocab
    -- in model returns tensor with 1 row per phrase, row is embedding vector of current word
    local x=LookupTable(params.vocab_size,params.dimension)(x_index);
    -- get graph nodes for computing next_h and next_c
    next_h,next_c=lstm(x,prev_h,prev_c)
    -- input table
    inputs={prev_h,prev_c,x_index};
    -- compile module
    local module= nn.gModule(inputs,{next_h,next_c});
    -- initialize weights to uniform distribution based on param
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    -- transfer module to gpu
    return module:cuda();
end


local function softmax_()
    -- inputs as identity nodes
    local y=nn.Identity()(); -- actual class
    local h_left=nn.Identity()();
    local h_right=nn.Identity()();
    -- linear layers with no bias for left and right lstm outputting values for each class
    local h2y_left= nn.Linear(params.dimension,5):noBias()(h_left)
    local h2y_right= nn.Linear(params.dimension,5):noBias()(h_right)
    -- add h2y_left and h2y_right outputs
    local h=nn.CAddTable()({h2y_left,h2y_right});
    -- apply softmax
    local pred= nn.LogSoftMax()(h) -- predicted class probabilities
    -- compute error
    local err= nn.ClassNLLCriterion()({pred, y})
    -- compile module
    local module= nn.gModule({h_left,h_right,y},{err,pred});

    -- init params to uniform distribution
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    -- transfer module to gpu
    return module:cuda()
end 

local function forward(Word,Word_r,Delete,isTraining)
    -- transfer word-index tensor to GPU
    Word=Word:cuda()
    -- for each word in the phrases
    for t=1,Word:size(2) do
        -- initialize h and c from the last batch (0's if it's the first batch)
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
            c_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_left[t-1];
            c_t_1=model.c_left[t-1];
        end

        -- prev_h, prev_c, t'th word of for each phrase
        inputs={h_t_1,c_t_1,Word:select(2,t)}

        -- during training dropout is used, during evaluate it's not
        if isTraining then
            model.lstms_left[t]:training();
        else
            model.lstms_left[t]:evaluate();
        end

        -- get next_h, next_c from model
        model.h_left[t],model.c_left[t]=unpack(model.lstms_left[t]:forward(inputs))

        -- if there are phrases in Words that have a zero for this column
        if Delete[t]:nDimension()~=0 then
            -- copy in the zeroes for those phrases
            model.h_left[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
            model.c_left[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end

    -- same as above but for reverse phrases
    Word=Word_r:cuda()
    for t=1,Word:size(2) do
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
            c_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_right[t-1];
            c_t_1=model.c_right[t-1];
        end
        inputs={h_t_1,c_t_1,Word_r:select(2,t)}
        if isTraining then
            model.lstms_right[t]:training();
        else
            model.lstms_right[t]:evaluate();
        end

        model.h_right[t],model.c_right[t]=unpack(model.lstms_right[t]:forward(inputs))
        if Delete[t]:nDimension()~=0 then
            model.h_right[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
            model.c_right[t]:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end
end

local function backward(Word,Delete,dh_left,dh_right)
    -- zero initial gradient for c
    local dc_left=torch.zeros(Word:size(1),params.dimension):cuda();
    -- iterate backwards through dimension
    for t=Word:size(2),1,-1 do

        -- get output from previous layer (input to this layer)
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
            c_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_left[t-1];
            c_t_1=model.c_left[t-1];
        end

        -- get gradients
        dh_left,dc_left=unpack(model.lstms_left[t]:backward({h_t_1,c_t_1,Word:select(2,t)},{dh_left,dc_left}));

        -- zero out gradients for phrases that aren't long enough for this index
        if Delete[t]:nDimension()~=0 then
            dh_left:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
            dc_left:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end

    -- same as above for reverse direction
    local dc_right=torch.zeros(Word:size(1),params.dimension):cuda();
    for t=Word:size(2),1,-1 do
        if t==1 then
            h_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
            c_t_1=torch.zeros(Word:size(1),params.dimension):cuda();
        else
            h_t_1=model.h_right[t-1];
            c_t_1=model.c_right[t-1];
        end
        dh_right,dc_right=unpack(model.lstms_right[t]:backward({h_t_1,c_t_1,Word:select(2,t)},{dh_right,dc_right}));
        if Delete[t]:nDimension()~=0 then
            dh_right:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
            dc_right:indexCopy(1,Delete[t],torch.zeros(Delete[t]:size(1),params.dimension):cuda())
        end
    end
end

local function test(filename)
    -- open file and set as default input
    open_train_file=io.open(filename,"r")
    io.input(open_train_file)
    local End,Y,Word,Word_r,Delete;

    End=0;
    local right=0;
    local total=0;
    -- loop through file in minibatches
    while End==0 do
        -- get indexed phrases
        End,Y,Word,Word_r,Delete=data.read_train(open_train_file,params.batch_size);
        Y=Y:cuda()
        if End==1 then 
            break;
        end
        -- forward through model for prediction
        forward(Word,Word_r,Delete,false);
        err,pred=unpack(softmax:forward({model.h_left[Word:size(2)],model.h_right[Word:size(2)],Y:cuda()}))

        -- get prediction
        local score,prediction=torch.max(pred,2)

        -- evaluate prediction against actual
        prediction:resize(prediction:size(1)*prediction:size(2));
        total=total+prediction:size(1);
        for i=1,prediction:size(1) do
            if Y[i]==prediction[i] then
                right=right+1;
            end
        end
    end

    -- return accuracy
    local accuracy=right/total;
    return accuracy;
end

cutorch.setDevice(1)
data=require("data")

-- get cuda lstms for left and right propogated networks
-- inputs: prev_h, prev_c and indexed sentence/phrase
-- output next_h and next_c
encoder_left =encoder_()
encoder_right =encoder_()

-- get cuda model for error calculation.
-- inputs: h_left, h_right, actual class
-- outputs: error, predicted class probabilities
softmax=softmax_();

-- make sure parameters are flattened
encoder_left:getParameters()
encoder_right:getParameters()
softmax:getParameters()

-- get params and gradients
paramx[1],paramdx[1]=encoder_left:parameters()
paramx[2],paramdx[2]=encoder_right:parameters()
paramx[3],paramdx[3] =softmax:parameters()

-- initialize ada - same shape as parameters with randomized epsilons
for i=1,3 do
    ada[i]={};
    for j=1,#paramx[i] do
        if paramx[i][j]:nDimension()==1 then
            ada[i][j]=1e-14*torch.rand(paramx[i][j]:size(1)):cuda()
        else
            ada[i][j]=1e-14*torch.rand(paramx[i][j]:size(1),paramx[i][j]:size(2)):cuda()
        end
    end
end

model.h_left={};
model.c_left={};
model.h_right={};
model.c_right={};

local timer=torch.Timer();
--embedding=data.read_embedding()

-- make left and right lstms use the same initial params
paramx[2][1]:copy(paramx[1][1])

-- get N copies of the lstm as array
model.lstms_left=g_cloneManyTimes(encoder_left,params.max_length)
model.lstms_right=g_cloneManyTimes(encoder_right,params.max_length)

iter=0;

-- store starting parameters
store_param={};
for i=1,#paramx do
    store_param[i]={}
    for j=1,#paramx[i] do
        store_param[i][j]=torch.Tensor(paramx[i][j]:size());
        store_param[i][j]:copy(paramx[i][j]);
    end
end
local best_accuracy=-1;

while iter<params.max_iter do
    iter=iter+1;

    -- open and set the training file as standard input
    open_train_file=io.open(params.train_file,"r")
    io.input(open_train_file)

    local End,Y,Word,Delete;
    End=0;
    local time1=timer:time().real;
    -- loop through train file in mini batches
    while End==0 do
        -- zero out gradients
        for i=1,#paramdx do
            for j=1,#paramdx[i] do
                paramdx[i][j]:zero();
            end
        end

        -- End = whether we finished reading the file
        -- Y = tensor of class labels
        -- Word/Word_r = tensor with word indices (forwards/reverse) at the end of each row
        -- Delete = for each column, indices of zeros (rows that don't have a word in that position)
        End,Y,Word,Word_r,Delete=data.read_train(open_train_file,params.batch_size);

        -- doesn't this throw out the last partial batch?
        if End==1 then 
            break;
        end

        -- evaluate each phrase in batch
        forward(Word,Word_r,Delete,true)

        -- get error and probs of predicted classes
        err,pred=unpack(softmax:forward({model.h_left[Word:size(2)],model.h_right[Word:size(2)],Y:cuda()}))

        -- get initial dh's
        local dh_left,dh_right=unpack(softmax:backward({model.h_left[Word:size(2)],model.h_right[Word:size(2)],Y:cuda()},{torch.ones(1):cuda(),torch.zeros(Word:size(1),5):cuda() }))

        -- backprop on main LSTMs
        backward(Word,Delete,dh_left,dh_right)

        -- update parameters using adagrad
        for i=1,#ada do
            for j=1,#ada[i] do
                if i==2 and j==1 then
                    paramdx[i][j]:add(paramdx[1][j]);
                end
                ada[i][j]:add(torch.cmul(paramdx[i][j],paramdx[i][j]))
                paramx[i][j]:add(-torch.cdiv(paramdx[i][j],torch.sqrt(ada[i][j])):mul(params.learning_rate))
            end
        end
        paramx[1][1]:copy(paramx[2][1])
    end
    local time2=timer:time().real;
    acc_dev=test(params.dev_file)
    acc_test=test(params.test_file)
    if acc_test>best_accuracy then
        best_accuracy=acc_test;
        for i=1,#paramx do
            for j=1,#paramx[i] do
                store_param[i][j]:copy(paramx[i][j]);
            end
        end
    end
    if iter==20 then
        break;
    end
end

print("test accuracy: "..best_accuracy)
-- output model to file storing best params found
local getNextFileName = require('../get-next-filename.lua')
local modelFileName = getNextFileName('model', '', '.', 2)
local file=torch.DiskFile(modelFileName,"w"):binary();
file:writeObject(store_param);
file:close();
