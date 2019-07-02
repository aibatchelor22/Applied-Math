clear all; close all; clc

%General plan for program
%Solve equation for a set of initial conditions
% Kuramoto-Sivashinsky equation (from Trefethen)
% u_t = -u*u_x - u_xx - u_xxxx,  periodic BCs 

N = 128;
x = 32*pi*(1:N)'/N;
input=[];
output=[];

for j=1:250
%Randomize initial boundary value function    
c1=rand/4; c2=rand/4; c3=rand/4; %Generate random harmonic coefficients
phi1=2*pi*(rand-0.5); phi2=2*pi*(rand-0.5); phi3=2*pi*(rand-0.5); %Generate random phases
u=c1*cos(2*pi*x/max(x(:))+phi1)+c2*cos(4*pi*x/max(x(:))+phi2)+c3*cos(6*pi*x/max(x(:))+phi3);
v = fft(u);    
%Spatial grid and initial condition:
h = 0.025;
k = [0:N/2-1 0 -N/2+1:-1]'/16;
L = k.^2 - k.^4;
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); 
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); 
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
%note nplt is divided by 250 in the original code.
tmax = 100; nmax = round(tmax/h); nplt = floor((tmax/250)/h); g = -0.5i*k;
for n = 1:nmax
t = n*h;
Nv = g.*fft(real(ifft(v)).^2);
a = E2.*v + Q.*Nv;
Na = g.*fft(real(ifft(a)).^2);
b = E2.*v + Q.*Na;
Nb = g.*fft(real(ifft(b)).^2);
c = E2.*a + Q.*(2*Nb-Nv);
Nc = g.*fft(real(ifft(c)).^2);
v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; if mod(n,nplt)==0
        u = real(ifft(v)); 
uu = [uu,u]; tt = [tt,t]; end
end


input=[input;uu(:,1:end-1).'];
output=[output;uu(:,2:end).'];

%input(:,:,j)=uu(:,1:end-1);
%output(:,:,j)=uu(:,2:end);

end



% figure(1)
% surf(tt,x,uu), shading interp, colormap(hot), axis tight 
% % view([-90 90]), colormap(autumn); 
% set(gca,'zlim',[-5 50]) 

%Train the neural net base on the random trajectories
net = feedforwardnet([60 60 60]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{1}.transferFcn = 'purelin';
net.trainParam.epochs=1000;
net = train(net,input.',output.','useGPU','yes','showResources','yes');

%Plot a new trajectory with new initial conditions
save(['KS_data.mat'],'x','tt','uu', 'input','output','-v7','net');

c1=rand/4; c2=rand/4; c3=rand/4; c4=rand/4; %Generate random harmonic coefficients
phi1=2*pi*(rand-0.5); phi2=2*pi*(rand-0.5); phi3=2*pi*(rand-0.5); phi4=2*pi*(rand-0.5); %Generate random phases
u=c1*cos(2*pi*x/max(x(:))+phi1)+c2*cos(4*pi*x/max(x(:))+phi2)+c3*cos(6*pi*x/max(x(:))+phi3);
uu=u;




v = fft(u);    
%Spatial grid and initial condition:
h = 0.025;
k = [0:N/2-1 0 -N/2+1:-1]'/16;
L = k.^2 - k.^4;
E = exp(h*L); E2 = exp(h*L/2);
M = 16;
r = exp(1i*pi*((1:M)-.5)/M);
LR = h*L(:,ones(M,1)) + r(ones(N,1),:);
Q = h*real(mean( (exp(LR/2)-1)./LR ,2)); 
f1 = h*real(mean( (-4-LR+exp(LR).*(4-3*LR+LR.^2))./LR.^3 ,2)); 
f2 = h*real(mean( (2+LR+exp(LR).*(-2+LR))./LR.^3 ,2));
f3 = h*real(mean( (-4-3*LR-LR.^2+exp(LR).*(4-LR))./LR.^3 ,2));

% Main time-stepping loop:
uu = u; tt = 0;
%note nplt is divided by 2500 in the original code.
tmax = 100; nmax = round(tmax/h); nplt = floor((tmax/250)/h); g = -0.5i*k;
for n = 1:nmax
t = n*h;
Nv = g.*fft(real(ifft(v)).^2);
a = E2.*v + Q.*Nv;
Na = g.*fft(real(ifft(a)).^2);
b = E2.*v + Q.*Na;
Nb = g.*fft(real(ifft(b)).^2);
c = E2.*a + Q.*(2*Nb-Nv);
Nc = g.*fft(real(ifft(c)).^2);
v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3; if mod(n,nplt)==0
        u = real(ifft(v)); 
uu = [uu,u]; tt = [tt,t]; end
end

unn=zeros(N,length(tt));
u=uu(:,1);
unn(:,1)=u;
for jj=2:length(tt)
    u0=net(u);
    unn(:,jj)=u0; u=u0;
end

figure(1)
surf(tt,x,uu), shading interp, colormap(hot), axis tight 
% view([-90 90]), colormap(autumn); 
set(gca,'zlim',[-5 50]) 

figure(2)
surf(tt,x,unn), shading interp, colormap(hot), axis tight 
% view([-90 90]), colormap(autumn); 
set(gca,'zlim',[-5 50]) 



