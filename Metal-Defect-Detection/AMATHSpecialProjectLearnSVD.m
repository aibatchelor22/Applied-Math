clear all; close all; clc

load('Defect1Raw.mat')
defectSet = im2double(dataSet);
load('NoDefect1Raw.mat')
noDefectSet = im2double(dataSet);
allData=[defectSet;noDefectSet];

[u,s,v] = svd(allData.', 'econ');

totalConfusionMatrix=zeros(2);
for numberOfTrials = 1:19
defectSet = v(1:50,:);
noDefectSet = v(51:100,:);
q1=randperm(50);
q2=randperm(50);
xtrain=[defectSet(q1(1:40),:); noDefectSet(q2(1:40),:)];
xtest=[defectSet(q1(41:end),:); noDefectSet(q2(41:end),:)];
ctrain=[ones(40,1); 2*ones(40,1)];
actual=[ones(10,1);2*ones(10,1)];
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
