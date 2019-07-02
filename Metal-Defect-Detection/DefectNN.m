clear all; close all; clc

load('Defect1Raw500.mat')
defectSet = im2double(dataSet);
load('NoDefect1Raw500.mat')
noDefectSet = im2double(dataSet);
datasize=500;

totalConfusionMatrix=zeros(2);
for numberOfTrials = 1:20
q1=randperm(datasize);
q2=randperm(datasize);
xtrain=[defectSet(q1(1:0.8*datasize),:); noDefectSet(q2(1:0.8*datasize),:)];
xtest=[defectSet(q1(0.8*datasize+1:end),:); noDefectSet(q2(0.8*datasize+1:end),:)];
ctrain=[-1*ones(0.8*datasize,1); ones(0.8*datasize,1)];
actual=[-1*ones(0.2*datasize,1);ones(0.2*datasize,1)];
%Apply Naive Bayes Classifier and test the model
% nb=fitcnb(xtrain,ctrain);
% pre=nb.predict(xtest);
%model = fitcensemble(xtrain,ctrain);
%pre=model.predict(xtest);
net=feedforwardnet([40 40 40]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net.trainParam.epochs=10000;
net = train(net,xtrain.',ctrain.','useGPU','yes','showResources','yes');
testTrial=net(xtest.');
testTrial=sign(testTrial);
currentConfusionMatrix = confusionmat(actual,testTrial);
totalConfusionMatrix = totalConfusionMatrix + currentConfusionMatrix;
end

totalConfusionMatrix = totalConfusionMatrix + currentConfusionMatrix;

% load handel;
% sound(y,Fs);
