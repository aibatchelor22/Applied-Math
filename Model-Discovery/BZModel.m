clear all; close all; clc;
load('BZ.mat');

t=1:1:1200;
dt=t(2)-t(1);
U=BZ_tensor(100:149,200:249,:); %Take  ROI for calculation
[m,n,k]=size(U); % x vs y vs time data

Udot=zeros(m,n,k-2);
for jj=1:m  % walk through x (space)
for jjj=1:n %walk through y (space)
for j=2:k-1  % walk through time
   Udot(jj,jjj,j-1)=( U(jj,jjj,j+1)-U(jj,jjj,j-1) )/(2*dt);
end
end
end

%Derivatives
dx=1;
dy=1;

%calculate Ux
Ux=zeros(m-2,n-2,k);
for time=1:k
for i=2:n-1
    for j=2:m-1
        Ux(j-1,i-1,time)=(U(j,i+1,time)-U(j,i-1,time))/2*dx;
    end
end
end

%calculate Uy
Uy=zeros(m-2,n-2,k);
for time=1:k
for i=2:n-1
    for j=2:m-1
        Uy(j-1,i-1,time)=(U(j+1,i,time)-U(j-1,i,time))/2*dy;
    end
end
end

%calculate Uxx
Uxx=zeros(m-2,n-2,k);
for time=1:k
for i=2:n-1
    for j=2:m-1
        Uxx(j-1,i-1,time)=(U(j,i+1,time)-2*U(j,i,time)+U(j,i-1,time))/dx.^2;
    end
end
end

%calculate Uyy
Uyy=zeros(m-2,n-2,k);
for time=1:k
for i=2:n-1
    for j=2:m-1
        Uyy(j-1,i-1,time)=(U(j+1,i,time)-2*U(j,i,time)+U(j-1,i,time))/dy.^2;
    end
end
end

%calculate Uxy
Uxy=zeros(m-2,n-2,k);
for time=1:k
for i=2:n-1
    for j=2:m-1
        Uxy(j-1,i-1,time)=(U(j+1,i+1,time)-U(j-1,i+1,time)-U(j+1,i-1,time)+U(j-1,i-1,time))/4*dx*dy;
    end
end
end

%Reshape all the function matrices, truncate where needed
u=reshape(U(2:49,2:49,2:1199),(n-2)*(m-2)*(k-2),1);
udot=reshape(Udot(2:49,2:49,:),(n-2)*(m-2)*(k-2),1);
ux=reshape(Ux(:,:,2:1199),(n-2)*(m-2)*(k-2),1);
uy=reshape(Uy(:,:,2:1199),(n-2)*(m-2)*(k-2),1);
uxx=reshape(Uxx(:,:,2:1199),(n-2)*(m-2)*(k-2),1);
uxy=reshape(Uxy(:,:,2:1199),(n-2)*(m-2)*(k-2),1);
uyy=reshape(Uyy(:,:,2:1199),(n-2)*(m-2)*(k-2),1);

%Define library of functions theta (noted here as "A")
zilch=zeros(2760192,1);
%A=[u u.^2 u.^3 ux uxx ux.*u ux.*ux ux.*uxx uy uyy uyy.*u uy.*uy uy.*uyy uxy uxy.*u];
A=[zilch zilch zilch ux zilch zilch zilch zilch uy zilch zilch zilch zilch zilch zilch];

%xi=A\udot;
xi=lasso(A,udot,'Lambda',0.002);
%xi=robustfit(A,udot,[],[],'off');
%xi=pinv(A)*udot;
bar(xi)

%Use selected functions from library and generate model prediction
Asparse=zeros(2760192,15);
%Asparse(:,1)=A(:,1);
Asparse(:,4)=A(:,4);
Asparse(:,9)=A(:,9);
%Asparse(:,10)=A(:,10);
Asparse(:,14)=A(:,14);
udotpredict=Asparse*xi;
