clear all; close all; clc;
clear all; close all; clc;
load('ridgeData.mat')
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

A = images;
B=zeros(10,60000);
for i = 1:60000
    currentVector = zeros(1,10);
    currentIndex = mod(labels(i),10);
    if currentIndex==0
        currentIndex=10;
    end
    currentVector(currentIndex) = 1;
    B(:,i)=currentVector;
end

%Solve for X
A = A.';
B=B.';

Xsparse=[];
maskMatrix=[];
for i = 1:10
%Find value for 90% of energy of weighting
Xslice=X(:,i);
sortedXslice = sort(abs(Xslice),'descend');
for j=1:784
    topCount=j;
    percent = sum(sortedXslice(1:j))/sum(sortedXslice);
    if percent > 0.9
        break
    end
end
[topPixels,I] = maxk(abs(Xslice),topCount);

maskVector = zeros(784,1);
for k = I
    maskVector(k)=1;
end
Xslicesparse=maskVector.*Xslice;
Xsparse=[Xsparse,Xslicesparse];
maskMatrix=[maskMatrix,maskVector];
end

%Make a sparse B from A and X
Bsparse=mtimes(A,Xsparse);

%Compare the sparse B with the original B
pixels = numel(Xsparse(Xsparse~=0));
error = immse(Bsparse,B);
BsparseRound=round(Bsparse);
Bdiff=B-BsparseRound;
trueValues=Bdiff(Bdiff>-0.5);
trueValues=trueValues(trueValues<0.5);
percentTrue=numel(trueValues)/numel(Bdiff);
Bfull=mtimes(A,X);
errorFull=immse(Bfull,B);
BfullRound=round(Bfull);
BdiffFull=B-BfullRound;
trueValuesFull=BdiffFull(BdiffFull>-0.5);
trueValuesFull=trueValuesFull(trueValuesFull<0.5);
percentTrueFull=numel(trueValuesFull)/numel(BdiffFull);

accurateCountFull=0;
for j=1:60000
[rowMaxPredicted,Ipredicted] = max(Bfull(j,:));
[rowMaxActual,Iactual] = max(B(j,:));
if Ipredicted==Iactual
  accurateCountFull = accurateCountFull + 1;
end
end
percentTrueFull=100*accurateCountFull/60000


accurateCountSparse=0;
for j=1:60000
[rowMaxPredicted,Ipredicted] = max(Bsparse(j,:));
[rowMaxActual,Iactual] = max(B(j,:));
if Ipredicted==Iactual
  accurateCountSparse = accurateCountSparse + 1;
end
end
percentTrueSparse=100*accurateCountSparse/60000