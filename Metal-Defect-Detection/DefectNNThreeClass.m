clear all; close all; clc

load('Defect1Raw200.mat')
defectSet1 = im2double(dataSet);
load('NoDefect1Raw200.mat')
noDefectSet1 = im2double(dataSet);
load('Defect2Raw200.mat')
defectSet2 = im2double(dataSet);
load('NoDefect2Raw200.mat')
noDefectSet2 = im2double(dataSet);
datasize=200;


totalFraction = 0;
%totalConfusionMatrix=zeros(2);
testRounds = 15;
for numberOfTrials = 1:testRounds
q1=randperm(datasize);
q2=randperm(datasize);
q3=randperm(datasize);
q4=randperm(datasize);
xtrain=[defectSet1(q1(1:0.8*datasize),:); noDefectSet1(q2(1:0.8*datasize),:);defectSet2(q3(1:0.8*datasize),:); noDefectSet2(q4(1:0.8*datasize),:)];
xtest=[defectSet1(q1(0.8*datasize+1:end),:); noDefectSet1(q2(0.8*datasize+1:end),:);defectSet2(q3(0.8*datasize+1:end),:); noDefectSet2(q4(0.8*datasize+1:end),:)];
ctrain=[ones(0.8*datasize,1); 2*ones(0.8*datasize,1);3*ones(0.8*datasize,1); 2*ones(0.8*datasize,1)];
actual=[ones(0.2*datasize,1); 2*ones(0.2*datasize,1);3*ones(0.2*datasize,1); 2*ones(0.2*datasize,1)];
%Apply Naive Bayes Classifier and test the model
% nb=fitcnb(xtrain,ctrain);
% pre=nb.predict(xtest);
%model = fitcensemble(xtrain,ctrain);
%pre=model.predict(xtest);
net=feedforwardnet(50);
%net.trainParam.lr = 0.5;
% net.layers{1}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'radbas';
% net.layers{3}.transferFcn = 'purelin';
net.trainParam.epochs=500;
net = train(net,xtrain.',ctrain.','useGPU','yes','showResources','yes');
testTrial=net(xtest.');
testTrial=round(testTrial);
correct=0;
for i = 1:numel(testTrial);
    if testTrial(i)==actual(i)
        correct = correct+1;
    end
end
currentFraction = correct/numel(testTrial);
totalFraction = totalFraction+currentFraction;

end

totalFraction = totalFraction / testRounds;
save(['Three_class_results.mat'],'totalFraction');
% load handel;
% sound(y,Fs);
