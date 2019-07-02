clear all; close all; clc

load('KaggleDefectWavelet.mat')
defectSet = im2double(dataSet);
load('KaggleNoDefectWavelet.mat')
noDefectSet = im2double(dataSet);
allData=[defectSet;noDefectSet];

[u,s,v] = svd(allData.', 'econ');

totalConfusionMatrix=zeros(2);
for numberOfTrials = 1:9
defectSet = v(1:20,:);
noDefectSet = v(21:50,:);
q1=randperm(20);
q2=randperm(30);
xtrain=[defectSet(q1(1:15),:); noDefectSet(q2(1:25),:)];
xtest=[defectSet(q1(16:end),:); noDefectSet(q2(26:end),:)];
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
% modes=[1,2,10,20,30,40,50,60,70,80,90,100];
% for j=1:12
% index=13;
% ff=u(:,1:modes(j))*s(1:modes(j),1:modes(j))*v(:,1:modes(j)).';
% testImageVector = ff(:,index);
% testImage = reshape(testImageVector,[480,640]);
% subplot(3,4,j);
% imagesc(testImage), colormap(gray), axis square, axis off
% end
