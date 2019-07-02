clear all; close all; clc

load('cam1_1.mat')
load('cam2_1.mat')
load('cam3_1.mat')

%load('cam1_2.mat')
%load('cam2_2.mat')
%load('cam3_2.mat')

%load('cam1_3.mat')
%load('cam2_3.mat')
%load('cam3_3.mat')

%load('cam1_4.mat')
%load('cam2_4.mat')
%load('cam3_4.mat')

videos = {'vidFrames1_1', 'vidFrames2_1', 'vidFrames3_1'};
%videos = {'vidFrames1_2', 'vidFrames2_2', 'vidFrames3_2'};
%videos = {'vidFrames1_3', 'vidFrames2_3', 'vidFrames3_3'};
%videos = {'vidFrames1_4', 'vidFrames2_4', 'vidFrames3_4'};

matrix=cell(3,1);
for cameraNumber=1:3
    currentVideo = videos{cameraNumber};
    vidFrames = evalin('base',currentVideo);
    numFrames = size(vidFrames, 4);
    for frameNumber = numFrames:-1:1
        mov(frameNumber).cdata = vidFrames(:,:,:,frameNumber);
        mov(frameNumber).colormap = [];      
    end
    clear diff
    XY_coord = zeros(1, numFrames-1);
    y_max=0; %track lowest value to synchronize videos
    for frameNumber=numFrames:-1:2
        currentFrame = rgb2gray(mov(frameNumber).cdata);
        previousFrame = rgb2gray(mov(frameNumber-1).cdata);
        difference(:,:,1) = imabsdiff(currentFrame, previousFrame);
        threshhold = graythresh(difference)*2;
        bw = (difference >= threshhold * 255);
        bw2 = bwareaopen(bw, 200); %remove all components that have fewer than 200 pixels
        s = regionprops(bwlabel(bw2(:,:,1)), 'centroid');
        c = [s.Centroid]; 
        x_avg=0;
        y_avg=0;
        count = 0;
        for j=1:numel(s)
            if s(j).Centroid(1) > 580
                continue
            end
            x_avg=s(j).Centroid(1)+x_avg;
            y_avg=s(j).Centroid(2)+y_avg;
            count = count + 1;
        end
        x_avg=x_avg/(count);
        y_avg=y_avg/(count);
        XY_coord(1, frameNumber-1) = x_avg;
        XY_coord(2, frameNumber-1) = y_avg;
        if y_avg > y_max
            y_max = y_avg;
        end
    end
    XY_coord(isnan(XY_coord)) = 0;
    XY_coord(2,:) = y_max - XY_coord(2,:);
    matrix{cameraNumber} = XY_coord;
end

% build final matrix from each vector

longest = cellfun('length', matrix);
longest = max(longest);
X = zeros(6, longest);

for n=1:3 %pad zeroes where needed
    temp_matrix = matrix{n,1};
    if (numel(temp_matrix(1,:)) < longest)
        temp_matrix = padarray(temp_matrix, [0 longest - numel(temp_matrix(1,:))], 'post');
    end
    matrix{n,1} = temp_matrix;
end

X = vertcat(matrix{1,1}, matrix{2,1}, matrix{3,1}); %The final matrix

[u,s,v]=svd(X); % perform SVD on the final matrix

semilogy(100*diag(s)/sum(diag(s)), 'ko','Linewidth',2) %plot the modes with respect to the diagonals of s
xlabel('Mode Number')
ylabel('Energy Percentage')
title('Horizontal Displacement and Rotation Case')
xticks([0:1:6])

%scatter(X(1,:),X(2,:))
%scatter(X(3,:),X(4,:))
%scatter(X(5,:),X(6,:))
%title('Camera A')
%title('Camera B')
%title('Camera C')
%xlabel('x pixel')
%ylabel('y pixel')