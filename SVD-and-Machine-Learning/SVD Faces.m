
%Part 1 load images and put them into a column vector

cd 'C:\Users\Ashley\Documents\MATLAB\Yale Faces Cropped\CroppedYale\';
faceData = dir('**/*.pgm');
facesArray = zeros(192*168,length(faceData));
for i = 1:numel(faceData)
filename = [faceData(i).folder '\' faceData(i).name];
currentImage = imread(filename); % read in image file to matrix
% resize images that are greater than 192x168
if faceData(i).bytes > 32271
    currentImage = imresize(currentImage, [192,168]);
end
facesColumn = reshape(currentImage, [], 1); %reshape 192x168 matrix to column vector
facesArray(:, i) = facesColumn; %put column vector into matrix
end

[u,s,v] = svd(facesArray, 'econ'); %do SVD on array of faces data

%Part 2 what is the interpretation of the U, Sigma, V?

%Part 3 What does the singular value spectrum look like and how many modes are necessary for good image reconstructions? (i.e. what is the rank r of the face space?)
semilogy(diag(s)/sum(diag(s)));

%images with various modes
modes = [1,5,10,100,200,300,500,600,700,800,900,1000];
for j=1:12
index=1979;
ff=u(:,1:modes(j))*s(1:modes(j),1:modes(j))*v(:,1:modes(j)).';
testImageVector = ff(:,index);
testImage = reshape(testImageVector,[192,168]);
subplot(3,4,j);
imagesc(testImage), colormap(gray), axis square, axis off
end
%Part 4 Compare the difference between the cropped (and aligned) versus uncropped images
%load the original images and do SVD
% matrixUncroppedIMG(:, i) = columnUncroppedIMG;
% [u1,s1,v1] = svd(matrixUncroppedIMG, 'econ');