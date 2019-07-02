clear all; close all; clc

z=[];

list= dir('C:\Users\Ashley\Documents\MATLAB\Kaggle No Defect\*');
dataSet = [];
for j=3:length(list)
    currentFileName = strcat('C:\Users\Ashley\Documents\MATLAB\Kaggle No Defect\', list(j).name);
    y = imread(currentFileName);
    [cA,cH,cV,cD] = dwt2(y,'sym4','mode','per');
    z1 = reshape(cA,[1,numel(cA)]);
    z2 = reshape(cH,[1,numel(cH)]);
    z3 = reshape(cV,[1,numel(cV)]);
    z4 = reshape(cD,[1,numel(cD)]);
    z = [z1,z2,z3,z4];
    dataSet = [dataSet;z];
    %dataSet = [dataSet;z];
end