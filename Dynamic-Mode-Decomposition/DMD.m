clear all; close all; clc;

load('Butterflies2.mat');
%[u,s,v] = svd(X,'econ');
numFrames=241;
frameRate = 30;
t=linspace(0,(numFrames/frameRate),numFrames); 
dt=t(2)-t(1);
X1 = X(:,1:end-1); X2 = X(:,2:end);
r=2;

[U, S, V] = svd(X1, 'econ');
r = min(r, size(U,2));
U_r = U(:, 1:r); % truncate to rank-r
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);
Atilde = U_r' * X2 * V_r / S_r; % low-rank dynamics
[W_r, D] = eig(Atilde);
Phi = X2 * V_r / S_r * W_r; % DMD modes
lambda = diag(D); % discrete-time eigenvalues
omega = log(lambda)/dt; % continuous-time eigenvalues
%% Compute DMD mode amplitudes b
x1 = X1(:, 1);
b = Phi\x1;

%% DMD reconstruction
time_dynamics = zeros(r, length(t));
for iter = 1:length(t),
time_dynamics(:,iter)=(b.*exp(omega*t(iter)));
end;
X_dmd_background = Phi*time_dynamics;
X_dmd_foreground = X - abs(X_dmd_background);
%Residual negative values
[m,n]=size(X_dmd_background);
R=zeros(m,n);
for i=1:m
    for j=1:n
        if X_dmd_background(m,n) > 0
            R(m,n)= X_dmd_background(m,n);
        end
    end
end
X_dmd_background = R + abs(X_dmd_background);
X_dmd_foreground = X_dmd_foreground - R;

X_dmd_background_real = real(X_dmd_background);
X_dmd_foreground_real = real(X_dmd_foreground);

%Use imshow to show a sample image