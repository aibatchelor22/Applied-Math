clear all; close all; clc

load('Defect1DWT.mat')
defectSet = im2double(dataSet);
load('NoDefect1DWT.mat')
noDefectSet = im2double(dataSet);



totalConfusionMatrix=zeros(2);
for numberOfTrials = 1:4
q1=randperm(50);
q2=randperm(50);
xtrain=[defectSet(q1(1:40),:); noDefectSet(q2(1:40),:)];
xtest=[defectSet(q1(41:end),:); noDefectSet(q2(41:end),:)];
ctrain=[ones(40,1); 2*ones(40,1)];
actual=[ones(10,1);2*ones(10,1)];
%Apply Naive Bayes Classifier and test the model
nb=fitcnb(xtrain,ctrain);
pre=nb.predict(xtest);
%model = fitcensemble(xtrain,ctrain);
%pre=model.predict(xtest);
currentConfusionMatrix = confusionmat(actual,pre);
totalConfusionMatrix = totalConfusionMatrix + currentConfusionMatrix;
end

totalConfusionMatrix = totalConfusionMatrix + currentConfusionMatrix;

load handel;
sound(y,Fs);
