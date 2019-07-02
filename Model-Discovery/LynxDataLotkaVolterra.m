clear all; close all; clc;

[~,~,data] = xlsread('HareLynx.csv');
hare = cell2mat(data(:,2));
lynx = cell2mat(data(:,3));

ti=(0:1:29)';
dt=0.1;
t=(0:dt:29)';


x1=hare;
x2=lynx;
x1=spline(ti,hare,t);
x2=spline(ti,lynx,t);

x0=[hare(1) lynx(1)];


n=length(t);
for j=2:n-1
  x1dot(j-1)=(x1(j+1)-x1(j-1))/(2*dt);
  x2dot(j-1)=(x2(j+1)-x2(j-1))/(2*dt);
end

one=ones(289,1);
zilch=zeros(289,1);
x1s=x1(2:n-1);
x2s=x2(2:n-1);

%A=[x1s x2s];
%A=[x1s x2s x1s.^2 x1s.*x2s x2s.^2 x1s.^3 (x1s.^2).*x2s (x2s.^2).*x1s x2s.^3 x1s.^4 x2s.^4 x1s.^5 x2s.^5 x1s.^-1 x2s.^-1];
%A=[x1s x2s zilch zilch zilch zilch zilch zilch x1s.^-1 x2s.^-1];

A1=[x1s zilch zilch x1s.*x2s zilch zilch zilch zilch zilch zilch];
A2=[zilch x2s zilch x1s.*x2s zilch zilch zilch zilch zilch zilch];

xi1=A1\x1dot.';
xi2=A2\x2dot.';
%lambda=0.002;
%xi1=lasso(A1,x1dot.','Lambda',lambda);
%xi2=lasso(A2,x2dot.','Lambda',lambda);
%xi1=robustfit(A1,x1dot,[],[],'off');
%xi2=robustfit(A2,x2dot,[],[],'off');

%xi1=pinv(A1)*x1dot.';
%xi2=pinv(A2)*x2dot.';
subplot(2,1,1), bar(xi1)
subplot(2,1,2), bar(xi2)

% xi1sparse=zeros(14,1);
% xi2sparse=zeros(14,1);
% xi1sparse(1)=xi1(1);
% %xi1sparse(2)=xi1(2);
% xi1sparse(4)=xi1(4);
% %xi1sparse(10)=xi1(10);
% %xi1sparse(11)=xi1(11);
% %xi1sparse(12)=xi1(12);
% %xi2sparse(1)=xi2(1);
% xi2sparse(2)=xi2(2);
% xi2sparse(4)=xi2(4);
% %xi2sparse(9)=xi2(9);
% %xi1sparse(10)=xi2(10);
% %xi1sparse(11)=xi2(11);
% %xi1sparse(12)=xi2(12);
% 
% x1dotpredict = A*xi1sparse;
% x2dotpredict = A*xi2sparse;



%Calculate Hankel Matrices
H1=[x1(1:100).'
   x1(6:105).'
   x1(11:110).'
   x1(16:115).'
   x1(21:120).'
   x1(26:125).'
   x1(31:130).'
   x1(36:135).'];

[u1,s1,v1]=svd(H1,'econ');
figure(2), subplot(2,1,1), plot(diag(s1)/(sum(diag(s1))),'ro','Linewidth',[3])

H2=[x2(1:100).'
   x2(6:105).'
   x2(11:110).'
   x2(16:115).'
   x2(21:120).'
   x2(26:125).'
   x2(31:130).'
   x2(36:135).'];

[u2,s2,v2]=svd(H2,'econ');
subplot(2,1,2), plot(diag(s2)/(sum(diag(s2))),'ro','Linewidth',[3])

sol = ode45(@(t,y) odefun(t,y,xi1,xi2),[t(1) t(numel(t))],[hare(1), lynx(1)]);
 x1predict=deval(sol,t,1);
 x2predict=deval(sol,t,2);
 
f=zeros(4,1);
lynxmean=mean(lynx);
haremean = mean(hare);
for i=1:numel(x1predict)
    if (x2predict(i) > lynxmean) && (x1predict(i) > haremean)
        f(1) = f(1) + 1;
    elseif (x2predict(i) > lynxmean) && (x1predict(i) < haremean)
        f(2) = f(2) + 1;
    elseif (x2predict(i) < lynxmean) && (x1predict(i) < haremean)
        f(3) = f(3) + 1;
    elseif (x2predict(i) < lynxmean) && (x1predict(i) > haremean)
        f(4) = f(4) + 1;
    end
end

g=zeros(4,1);

for i=1:numel(lynx)
    if ((lynx(i) > lynxmean) && (hare(i) > haremean))
        g(1) = g(1) + 1;
    elseif ((lynx(i) > lynxmean) && (hare(i) < haremean))
        g(2) = g(2) + 1;
    elseif ((lynx(i) < lynxmean) && (hare(i) < haremean))
        g(3) = g(3) + 1;
    elseif ((lynx(i) < lynxmean) && (hare(i) > haremean))
        g(4) = g(4) + 1;
    end
end

f=f/sum(f);
g=g/sum(g);
Int=f.*log(f./g); % compute integrand
Int(isinf(Int))=0; Int(isnan(Int))=0;
KL=sum(Int); % KL divergence

parameters=2;
n=numel(x1);
varianceLynx=(1/n)*(sum((x2predict(:)-x2(:)).^2))
varianceHare=(1/n)*(sum((x1predict(:)-x1(:)).^2))
AICLynx=2*parameters+n*log(2*pi)+n*log(varianceLynx)+n
BICLynx=log(n)*parameters+n*log(2*pi)+n*log(varianceLynx)+n
AICHare=2*parameters+n*log(2*pi)+n*log(varianceHare)+n
BICHare=log(n)*parameters+n*log(2*pi)+n*log(varianceHare)+n



 function odes = odefun(t,y,xi1,xi2)
 
 
 ode1 = xi1(1)*y(1) + xi1(4)*y(1)*y(2);
 ode2 = xi2(2)*y(2) + xi2(4)*y(1)*y(2);
 odes = [ode1; ode2];
 end