clear all; close all; clc

load('KS_data.mat');

net = feedforwardnet([5 5 5]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');

%Plot a new trajectory with new initial conditions

c1=rand/4; c2=rand/4; c3=rand/4; %Generate random harmonic coefficients
phi1=2*pi*(rand-0.5); phi2=2*pi*(rand-0.5); phi3=2*pi*(rand-0.5); %Generate random phases
u=c1*cos(x/16+phi1)+c2*cos(x/8+phi2)+c3*cos(x/4+phi3);

unn=zeros(25,128);
unn(1,:)=u;
for jj=2:length(tt)
    u0=net(u);
    unn(jj,:)=u0.'; u=u0;
end