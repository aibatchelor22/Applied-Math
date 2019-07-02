clear all; close all; clc

z=[];

list= dir('C:\Users\Ashley\Documents\MATLAB\Kaggle Defect\*');
dataSet = [];
for j=3:length(list)
    currentFileName = strcat('C:\Users\Ashley\Documents\MATLAB\Kaggle Defect\', list(j).name);
    y = imread(currentFileName);
    waveletY = dwt2(y,'sym4','mode','per');;
    dataSet = [dataSet,waveletY];
end