clear all, close all

% Simulate Lorenz system
dt=0.01; T=8; t=0:dt:T;
b=8/3; sig=10; rho=[10,28,40];

testrho=17;


input=[]; output=[];
inputentry=zeros(800,4);
for i=1:3
    r=rho(i);
for j=1:100  % training trajectories
    x0=30*(rand(3,1)-0.5);
    Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  r * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
    [t,y] = ode45(Lorenz,t,x0);
    inputentry(:,1:3)=y(1:end-1,:);
    inputentry(:,4)=rho(i);
    input=[input; inputentry];
    output=[output; y(2:end,:)];
    %plot3(y(:,1),y(:,2),y(:,3)), hold on
    %plot3(x0(1),x0(2),x0(3),'ro')
end
end
%grid on, view(-23,18)


%%
net = feedforwardnet([20 20 20]);
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'radbas';
net.layers{3}.transferFcn = 'purelin';
net = train(net,input.',output.');


%%
figure(2)
x0=20*(rand(3,1)-0.5);
Lorenz = @(t,x)([ sig * (x(2) - x(1))       ; ...
                  testrho * x(1)-x(1) * x(3) - x(2) ; ...
                  x(1) * x(2) - b*x(3)         ]);              
    ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);
    [t,y] = ode45(Lorenz,t,x0);
[t,y] = ode45(Lorenz,t,x0);
plot3(y(:,1),y(:,2),y(:,3)), hold on
plot3(x0(1),x0(2),x0(3),'ro','Linewidth',[2])
grid on

state=zeros(4,1);
ynn(1,:)=x0;
for jj=2:length(t)
    state(1:3)=x0;
    state(4)=testrho;
    y0=net(state);
    ynn(jj,:)=y0.'; x0=y0;
end
plot3(ynn(:,1),ynn(:,2),ynn(:,3),':','Linewidth',[2])

figure(3)
subplot(3,2,1), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
subplot(3,2,3), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
subplot(3,2,5), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])


figure(3)
subplot(3,2,2), plot(t,y(:,1),t,ynn(:,1),'Linewidth',[2])
subplot(3,2,4), plot(t,y(:,2),t,ynn(:,2),'Linewidth',[2])
subplot(3,2,6), plot(t,y(:,3),t,ynn(:,3),'Linewidth',[2])

%%
figure(2), view(-75,15)
figure(3)
subplot(3,2,1), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,2), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,3), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,4), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,5), set(gca,'Fontsize',[15],'Xlim',[0 8])
subplot(3,2,6), set(gca,'Fontsize',[15],'Xlim',[0 8])
legend('Lorenz','NN')