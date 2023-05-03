%% Model4 2维CNN空间+lstm算法
imageInputSize = [7 5 5]; % input size
numClasses = 1; % num of classes, because this model is regression model, this parameter is 1
filterSize1 = [3,3]; % kernel size of CNN layer
numFilters1 = 30; % kernel number of CNN layer
filterSize2 = [5,3]; % kernel size of CNN layer
numFilters2 = 30; % kernel number of CNN layer
numHiddenUnits1=100; % num of units in hidden layer1
numHiddenUnits2=100; % num of units in hidden layer2
numHiddenUnits3=100; % num of units in hidden layer3
layers = [
    sequenceInputLayer(imageInputSize,'Normalization','none','Name','Input')
    sequenceFoldingLayer('Name','fold')
    batchNormalizationLayer('Name','bn1')
    convolution2dLayer(filterSize1,numFilters1,'Name','conv1')% CNN layer
    reluLayer('Name','relu1')
    convolution2dLayer(filterSize2,numFilters2,'Name','conv2')% CNN layer
    reluLayer('Name','relu2')
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')

    % 3-layer BiLSTM
    bilstmLayer(numHiddenUnits1, 'OutputMode', 'sequence','Name','lstm1')
    dropoutLayer(0.25,'Name','drop1')
    bilstmLayer(numHiddenUnits2, 'OutputMode', 'sequence','Name','lstm2')
    dropoutLayer(0.25,'Name','drop2')
    bilstmLayer(numHiddenUnits3, 'OutputMode', 'last','Name','lstm3')
    dropoutLayer(0.25,'Name','drop3')

    % dense network
    fullyConnectedLayer(100,'Name','fc1')
    dropoutLayer(0.25,'Name','drop4')
    reluLayer('Name','relu5')
    fullyConnectedLayer(50,'Name','fc2')
    reluLayer('Name','relu6')
    fullyConnectedLayer(numClasses,'Name','fc3')
    regressionLayer('Name','Output')];% model structure

lgraph = layerGraph(layers);
lgraph= connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize'); %connect the subnetworks
analyzeNetwork(lgraph); % output the info of model
%% Data loading and processing 数据载入和处理
% Data loading
load task_data4class
load rest_data4class
load rest0_fepoch4class
% Data reshuffling and integretting
rest1ind=randperm(2724);
rest2ind=randperm(2730);
task_fe=[task_fe;rest_fe;rest1_fe(rest1ind(1:500),:,:,:);rest2_fe(rest2ind(1:500),:,:,:)];
%axises of data: 1-sample, 2-time, 3-channel, 4-frequency band
%1000 no-vrms data samples are involved in the dataset
task_ssq=[task_ssq;rest_ssq;zeros(500,4);zeros(500,4)];% labels

% data and label normalization
task_fe=(task_fe-mean(task_fe,2))./std(task_fe,[],2);
minssq=min(task_ssq,[],1);
maxssq=max(task_ssq,[],1);
norm_ssq=task_ssq./maxssq;

% permutation for the sequece input of LSTM or BiLSTM, 
% the time axis should be the last axis of each sample
task_fe=permute(task_fe,[3,4,2,1]);

% space mapping for electrodes
task_fe2=zeros(7,5,5,200,7335);
task_fe2(1,[1,2,4,5],:,:,:)=task_fe([3,1,2,9],:,:,:);
task_fe2(2,:,:,:,:)=task_fe([4,5,6,7,8],:,:,:);
task_fe2(3,:,:,:,:)=task_fe([10,11,12,13,14],:,:,:);
task_fe2(4,:,:,:,:)=task_fe([15,16,17,18,19],:,:,:);
task_fe2(5,[2,3,4],:,:,:)=task_fe([20,21,22],:,:,:);
task_fe2(6,:,:,:,:)=task_fe([23,24,25,26,27],:,:,:);
task_fe2(7,[2,3,4],:,:,:)=task_fe([28,29,30],:,:,:);
task_fe=task_fe2;
%% Training
all_ytest=[];
all_ypred=[];
indices=crossvalind('Kfold',task_ssq(:,4),10);
% 10-fold cross-validation
for i=1:10
    test=(indices==i);
    train=~test;

    %train set and test set
    XTrain=task_fe(:,:,:,:,train==true);
    XTest=task_fe(:,:,:,:,test);
    YTrain=norm_ssq(train,4);
    YTest=task_ssq(test,4);

    %train set reshuffling for each fold and validation set extraction
    trainnum=size(YTrain,1);
    randind=randperm(trainnum);
    YVal=YTrain(randind(1:fix(0.2*trainnum)));
    YTrain=YTrain(randind(fix(0.2*trainnum)+1:end));
    XVal=XTrain(:,:,:,:,randind(1:fix(0.2*trainnum)));
    XTrain=XTrain(:,:,:,:,randind(fix(0.2*trainnum)+1:end));
    
    %adjusting data set for the training of LSTM or BiLSTM
    XTrain=squeeze(mat2cell(XTrain,size(XTrain,1),size(XTrain,2),size(XTrain,3),size(XTrain,4),ones(1,size(XTrain,5))));
    XVal=squeeze(mat2cell(XVal,size(XVal,1),size(XVal,2),size(XVal,3),size(XVal,4),ones(1,size(XVal,5))));
    XTest=squeeze(mat2cell(XTest,size(XTest,1),size(XTest,2),size(XTest,3),size(XTest,4),ones(1,size(XTest,5))));

    %training sets
    options = trainingOptions('adam', ...
        'MaxEpochs',50,...
        'ValidationData',{XVal,YVal},...
        'ValidationFrequency',30,...
        'ValidationPatience',20,...
        'InitialLearnRate',1e-4, ...
        'Verbose',false, ...
        'MiniBatchSize',32, ...
        'ExecutionEnvironment','gpu',...
        'Plots','training-progress');

    %training model
    [net{i},info{i}]= trainNetwork(XTrain, YTrain,lgraph,options);
    %giving the prediction of the test set
    YPred = predict(net{i},XTest);
    YPred=double(YPred);
    % Antinormalization
    YPred=YPred*maxssq(4);

    % compute the MSE between real score and predict score
    loss(i)=mean((YPred-YTest).^2);

    all_ytest=[all_ytest;YTest];
    all_ypred=[all_ypred;YPred];
end
% Plot the regression results of all test sets
figure;plotregression(all_ytest,all_ypred,'Regression')
% Save the results
save cv_m4_3bilstm100_result net info loss all_ytest all_ypred indices

