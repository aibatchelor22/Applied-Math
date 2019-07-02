clear all, close all

% Simulate Lorenz system
dt=0.01; T=20; t=0:dt:T;
b=8/3; sig=10; %rho=[10,28,40];

input=[]; output=[];
outputentry=zeros(numel(t(:)),6);


    r=28;
for j=1:300  % training trajectories
    x0=30*(rand(3,1)-0.5);
    Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
    [t,y] = ode45(Lorenz,t,x0);
    outputentry(:,1:3)=y(1:end,:);
    outputentry(1,4)=sign(x0(1));
    outputentry(1,5)=2;
    for k=2:numel(t(:))-1
      outputentry(k,4)=sign(outputentry(k,1));
      outputentry(k,5)=outputentry(k,4)-outputentry(k-1,4);
    end
    outputentry(numel(t(:))-1,6)=0;
    for k=numel(t(:))-2:-1:1
      if abs(outputentry(k,5))==2
          outputentry(k,6)=0;
      else
          outputentry(k,6)=outputentry(k+1,6)+1;
      end
    end
    for k=numel(t(:))-2:-1:1
      if abs(outputentry(k,5))==2
        trajectorysize=k-1;
        break
      end
    end
    inputentry=zeros(trajectorysize,4);
    outputentryadjusted=zeros(trajectorysize,1);
    inputentry=y(2:trajectorysize+1,:);
%     outputentryadjusted(:,1)=outputentry(2:trajectorysize+1,1);
%     outputentryadjusted(:,2)=outputentry(2:trajectorysize+1,2);
%     outputentryadjusted(:,3)=outputentry(2:trajectorysize+1,3);
    outputentryadjusted(:,1)=outputentry(2:trajectorysize+1,6);
    input=[input; inputentry];
    output=[output; outputentryadjusted];
    %plot3(y(:,1),y(:,2),y(:,3)), hold on
    %plot3(x0(1),x0(2),x0(3),'ro')
end

%grid on, view(-23,18)



net = feedforwardnet(60);
% net.layers{1}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'radbas';
% net.layers{3}.transferFcn = 'radbas';
net.trainParam.epochs=10000;
net = train(net,input.',output.','useGPU','yes','showResources','yes');


figure(2)
x0=30*(rand(3,1)-0.5);
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
[t,y] = ode45(Lorenz,t,x0);
% plot3(y(:,1),y(:,2),y(:,3)), hold on
% plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
% grid on


for jj=1:length(t)
    x0=y(jj,:);
    y0=net(x0.');
    ytimer(jj,:)=y0;
end

ytimeActualEntry=zeros(numel(t(:)),6);
ytimeActualEntry(:,1:3)=y(1:end,:);
    ytimeActualEntry(1,4)=sign(x0(1));
    ytimeActualEntry(1,5)=2;
    for k=2:numel(t(:))-1
      ytimeActualEntry(k,4)=sign(ytimeActualEntry(k,1));
      ytimeActualEntry(k,5)=ytimeActualEntry(k,4)-ytimeActualEntry(k-1,4);
    end
    ytimeActualEntry(numel(t(:))-1,6)=0;
    for k=numel(t(:))-2:-1:1
      if abs(ytimeActualEntry(k,5))==2
          ytimeActualEntry(k,6)=0;
      else
          ytimeActualEntry(k,6)=ytimeActualEntry(k+1,6)+1;
      end
    end
    for k=numel(t(:))-2:-1:1
      if abs(ytimeActualEntry(k,5))==2
        trajectorysize=k-1;
        break
      end
    end
yTimeActual=zeros(trajectorysize,1);
yTimeActual(:,1)=outputentry(2:trajectorysize+1,6);
plot(t(1:numel(yTimeActual)),yTimeActual);
hold on
plot(t(1:numel(yTimeActual)),ytimer(1:numel(yTimeActual)));