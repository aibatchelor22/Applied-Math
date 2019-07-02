clear all; close all; clc

load('KaggleDefectDWT.mat')
defectSet = im2double(dataSet);
load('KaggleNoDefectDWT.mat')
noDefectSet = im2double(dataSet);



totalConfusionMatrix=zeros(2);
for numberOfTrials = 1:19
q1=randperm(20);
q2=randperm(30);
xtrain=[defectSet(q1(1:15),:); noDefectSet((q2(1:25)+20),:)];
xtest=[defectSet(q1(16:end),:); noDefectSet((q2(26:end)+20),:)];
ctrain=[ones(15,1); 2*ones(25,1)];
actual=[ones(5,1);2*ones(5,1)];
%Apply Naive Bayes Classifier and test the model
nb=fitcnb(xtrain,ctrain);
pre=nb.predict(xtest);
%model = fitcsvm(xtrain,ctrain);
%pre=model.predict(xtest);
currentConfusionMatrix = confusionmat(actual,pre);
totalConfusionMatrix = totalConfusionMatrix + currentConfusionMatrix;
end

totalConfusionMatrix = totalConfusionMatrix + currentConfusionMatrix;

load handel;
sound(y,Fs);
