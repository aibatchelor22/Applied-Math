clear all; close all; clc;
%http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
A = images;
B=zeros(10,60000);
digit=[1,0,0,0,0,0,0,0,0,0];
Asingle = [];
Bsingle = [];
for i = 1:60000
    currentVector = zeros(1,10);
    currentIndex = mod(labels(i),10);
    if currentIndex==0
        currentIndex=10;
    end
    currentVector(currentIndex) = 1;
    B(:,i)=currentVector;
    if currentVector==digit
        Asingle = [Asingle, A(:,i)];
        Bsingle = [Bsingle, B(:,i)];
    end
end


%Solve for X
Asingle = Asingle.';
Bsingle=Bsingle.';
%Backslash
%X=A\B;

%Pseudoinverse
Xval=pinv(Asingle)*Bsingle;

%robustfit
%Xval = robustfit(A,B);

%LASSO
%lambda=0.1;
%[X,stats]=lasso(A,B,'Lambda',lambda);

%Ridge
%X=ridge(B,A,0.5,0);

%Plot loading values of X with pcolor

rowSum = zeros(784,1);
%Plot histogram of loading values

%pcolor of loading values
%h = pcolor(XSparse);
%set(h, 'EdgeColor', 'none');
%pcolor of rowSum
%compositeImage=reshape(rowSum,[28,28]);
%pcolor(compositeImage)

%Sum and rank the rows of X
for j=1:784
    rowSum(j)=sum(abs(Xval(j,:)));
end


%Sort and display top rows
sortedRows=sort(abs(rowSum),'descend');
bar(sortedRows/sum(sortedRows(:)));
[topRows,I] = maxk(rowSum,50);
I=I.';

maskMatrix = zeros(784,10);
for k = I
    maskMatrix(k,:)=1;
end
XSparse=maskMatrix.*Xval;