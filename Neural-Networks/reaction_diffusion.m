clear all; close all; clc

% lambda-omega reaction-diffusion system
%  u_t = lam(A) u - ome(A) v + d1*(u_xx + u_yy) = 0
%  v_t = ome(A) u + lam(A) v + d2*(v_xx + v_yy) = 0
%
%  A^2 = u^2 + v^2 and
%  lam(A) = 1 - A^2
%  ome(A) = -beta*A^2


t=0:0.05:10;
d1=0.1; d2=0.1; beta=1.0;
L=20; n=64; N=n*n;
x2=linspace(-L/2,L/2,n+1); x=x2(1:n); y=x;
kx=(2*pi/L)*[0:(n/2-1) -n/2:-1]; ky=kx;
trajectories=101;

allData=[];

for i=1:trajectories-1

% INITIAL CONDITIONS

[X,Y]=meshgrid(x,y);
[KX,KY]=meshgrid(kx,ky);
K2=KX.^2+KY.^2; K22=reshape(K2,N,1);

%m=1; % number of spirals

u = zeros(length(x),length(y),length(t));
v = zeros(length(x),length(y),length(t));

c1=rand; c2=rand; c3=rand; %Generate random harmonic coefficients
c4=rand; c5=rand; c6=rand;
phi1=2*pi*(rand-0.5); phi2=2*pi*(rand-0.5); phi3=2*pi*(rand-0.5); 
phi4=2*pi*(rand-0.5); phi5=2*pi*(rand-0.5); phi6=2*pi*(rand-0.5);
u(:,:,1)=c1*cos(4*X/L+phi1)+c2*cos(2*X/L+phi2)+c3*cos(3*X/L+phi3)+...
    c4*cos(4*Y/L+phi4)+c5*cos(2*Y/L+phi5)+c6*cos(3*Y/L+phi6);

c1=rand; c2=rand; c3=rand;
c4=rand; c5=rand; c6=rand; %Generate random harmonic coefficients
phi4=2*pi*(rand-0.5); phi5=2*pi*(rand-0.5); phi6=2*pi*(rand-0.5);
v(:,:,1)=c1*cos(4*X/L+phi1)+c2*cos(2*X/L+phi2)+c3*cos(3*X/L+phi3)+...
    c4*cos(4*Y/L+phi4)+c5*cos(2*Y/L+phi5)+c6*cos(3*Y/L+phi6);

% REACTION-DIFFUSION
uvt=[reshape(fft2(u(:,:,1)),1,N) reshape(fft2(v(:,:,1)),1,N)].';
[t,uvsol]=ode45('reaction_diffusion_rhs',t,uvt,[],K22,d1,d2,beta,n,N);


for j=1:length(t)-1
ut=reshape((uvsol(j,1:N).'),n,n);
vt=reshape((uvsol(j,(N+1):(2*N)).'),n,n);
u(:,:,j+1)=real(ifft2(ut));
v(:,:,j+1)=real(ifft2(vt));

%figure(1)
%pcolor(x,y,v(:,:,j+1)); shading interp; colormap(hot); colorbar; drawnow; 
end

totalFrameInstance=[];
trajData=[];
%reshape u into a vector
for k=1:length(t);
    uFrame=squeeze(u(:,:,k));
    uVector=reshape(uFrame,[1,4096]);
    vFrame=squeeze(v(:,:,k));
    vVector=reshape(vFrame,[1,4096]);
    totalFrameInstance=[uVector,vVector];
    trajData=[trajData; totalFrameInstance];
end
%reshape u and v into vectors
allData=[allData;trajData];
% uInputentry=uData(1:end-numel(uVector));
% uOutputentry=uData(numel(uVector):end);
% vInputentry=vData(1:end-numel(vVector));
% vOutputentry=vData(numel(vVector):end);

% inputentry(:,:,:,2)=v(:,:,1:end-1);
% inputentry(:,:,:,1)=u(:,:,2:end);
% inputentry(:,:,:,2)=v(:,:,2:end);
% input=[input; inputentry];
% output=[output; outputentry];

end

% %Take svd of data matrix
allData=allData.';
[yu, s, vee] = svd(allData,'econ');
% 
% %plot s matrix diagonals
%plot(diag(s)/sum(diag(s)),'o');

rank = 18;
%reducedData = sr*vee; %FIX THIS!
reducedData=s(1:rank,1:rank)*vee(:,1:rank).';
input=[];
output=[];
for i=1:trajectories-1
    inputentry=reducedData(:,(i-1)*length(t)+1:i*length(t)-1);
    input=[input; inputentry.'];
    outputentry=reducedData(:,(i-1)*length(t)+2:i*length(t));
    output=[output; outputentry.'];
end



%Predict trajectory of low rank variables
net = feedforwardnet([50 50 50]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net.trainParam.epochs=10000;
net = train(net,input.',output.','useGPU','yes','showResources','yes');


c1=rand; c2=rand; c3=rand;
c4=rand; c5=rand; c6=rand; %Generate random harmonic coefficients
phi1=2*pi*(rand-0.5); phi2=2*pi*(rand-0.5); phi3=2*pi*(rand-0.5);
phi4=2*pi*(rand-0.5); phi5=2*pi*(rand-0.5); phi6=2*pi*(rand-0.5);
initialu=c1*cos(4*X/L+phi1)+c2*cos(2*X/L+phi2)+c3*cos(3*X/L+phi3)+...
    c4*cos(4*Y/L+phi4)+c5*cos(2*Y/L+phi5)+c6*cos(3*Y/L+phi6);
c1=rand; c2=rand; c3=rand;
c4=rand; c5=rand; c6=rand; %Generate random harmonic coefficients
phi1=2*pi*(rand-0.5); phi2=2*pi*(rand-0.5); phi3=2*pi*(rand-0.5);
phi4=2*pi*(rand-0.5); phi5=2*pi*(rand-0.5); phi6=2*pi*(rand-0.5);
initialv=c1*cos(4*X/L+phi1)+c2*cos(2*X/L+phi2)+c3*cos(3*X/L+phi3)+...
    c4*cos(4*Y/L+phi4)+c5*cos(2*Y/L+phi5)+c6*cos(3*Y/L+phi6);
%uFrame=squeeze(u(:,:,1));
uVector=reshape(initialu,[1,4096]);
%vFrame=squeeze(v(:,:,1));
vVector=reshape(initialv,[1,4096]);
testState=[uVector,vVector];
testStateMatrix=zeros(trajectories*length(t),length(testState));
testStateMatrix(1,:)=testState;
yu_r=yu(:,1:rank);
%lowRankTestStateMatrix=yu_r.'*testStateMatrix;
initialtestState=yu_r.'*testState.'; %project test state into low rank basis

trajNN=zeros(rank,length(t));
trajNN(:,1)=initialtestState;
trajNext=trajNN(:,1);
for jj=2:length(t)
    traj0=net(trajNext);
    trajNN(:,jj)=traj0.'; trajNext=traj0;
end

%Reshape low rank trajectories back into u and v
originalbasisDataNN=yu_r*trajNN; 
uNNFrames=originalbasisDataNN(1:4096,:);
vNNFrames=originalbasisDataNN(4097:8192,:);
uNN=zeros(64,64,length(t));
vNN=zeros(64,64,length(t));
for i=1:length(t)
    uNN(:,:,i)=reshape(uNNFrames(:,i),[64,64]);
    vNN(:,:,i)=reshape(vNNFrames(:,i),[64,64]);
end
% uNN=reshape(uNNFrames,[64,64,length(t)]);
% vNN=reshape(vNNFrames,[64,64,length(t)]);

%Plot low rank trajectories

%Generate ode45 test trajectory
u = zeros(length(x),length(y),length(t));
v = zeros(length(x),length(y),length(t));
u(:,:,1)=initialu;
v(:,:,1)=initialv;

% REACTION-DIFFUSION
uvt=[reshape(fft2(u(:,:,1)),1,N) reshape(fft2(v(:,:,1)),1,N)].';
[t,uvsol]=ode45('reaction_diffusion_rhs',t,uvt,[],K22,d1,d2,beta,n,N);


for j=1:length(t)-1
ut=reshape((uvsol(j,1:N).'),n,n);
vt=reshape((uvsol(j,(N+1):(2*N)).'),n,n);
u(:,:,j+1)=real(ifft2(ut));
v(:,:,j+1)=real(ifft2(vt));
end

figure(1)
pcolor(u(:,:,100)), shading interp
figure(2)
pcolor(uNN(:,:,100)),shading interp

save('reaction_diffusion_big.mat','t','x','y','u','v','yu','s','vee','rank','input','output')

%%
%load reaction_diffusion_big
%pcolor(x,y,u(:,:,end)); shading interp; colormap(hot)