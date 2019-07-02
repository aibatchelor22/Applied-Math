clear all; close all; clc;
%http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
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
X=[];

for j=1:10
%Make B a vector
Bsingle=B(:,j);

%Backslash
%Xsingle=A\Bsingle;

%Pseudoinverse
%Xsingle=pinv(A)*Bsingle;

%LASSO
lambda=0.05;
Xsingle=lasso(A,Bsingle,'Lambda',lambda);

%Ridges
%Xsingle=ridge(Bsingle,A,0.05);

%Robustfit
%Xsingle=robustfit(A,Bsingle,[],[],'off');

X=[X,Xsingle];
end

%Plot loading values of X with pcolor

rowSum = zeros(784,1);
%Sum and rank the rows of X
for j=1:784
    rowSum(j)=sum(abs(X(j,:)));
end

%Plot histogram of loading values


%pcolor of rowSum
%compositeImage=reshape(rowSum,[28,28]);
%pcolor(abs(compositeImage)), shading interp, colormap(hot)


%Sort and display top rows
topCount=95;
sortedRows=sort(abs(rowSum),'descend');
bar(sortedRows/sum(sortedRows(:)));
[topRows,I] = maxk(rowSum,topCount);
I=I.';

maskMatrix = zeros(784,10);
for k = I
    maskMatrix(k,:)=1;
end
Xsparse=maskMatrix.*X;

%pcolor of loading values
%h = pcolor(XSparse);
%set(h, 'EdgeColor', 'none');

%Make a sparse B from A and X
Bsparse=mtimes(A,Xsparse);

%Compare the sparse B with the original B, box()?
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


