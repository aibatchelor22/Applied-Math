clear all; close all; clc;
load Testdata
L=15; % spatial domain
n=64; % Fourier modes
x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x;
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; ks=fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

UnAve = zeros(n,n,n)

%Take a fourier transform of each image and average them in the Fourier
%Domain, store the fourier transformed images and the spatial domain images
%in an array for later
for j=1:20
  Un(:,:,:)=reshape(Undata(j,:),n,n,n);
  Uninstance(j,:,:,:)=Un(:,:,:);
  Ut=fftshift(fftn(Un));
  Utinstance(j,:,:,:)=Ut(:,:,:);
  UnAve = UnAve + Ut;
end
UnAveAbs = abs(UnAve);
UnAveAbs = UnAveAbs / max(UnAveAbs(:));

%Find the max of the averaged spectra
[C,I] = max(UnAveAbs(:));
[I1,I2,I3] = ind2sub(size(UnAveAbs),I);

%Quick check of a slice of the Fourier Domain
UnAveAbsSlice = squeeze(UnAveAbs(:,:,I3));
pcolor(ks,ks,UnAveAbsSlice);

%Create a filter around the max value of the averaged spectrum
KxFilt = 1.885;%Kx(I1);
KyFilt = -1.0472;%Ky(I2);
KzFilt = 0;%Kz(I3);
filter = exp(-0.2*((Kx-KxFilt).^2+(Ky-KyFilt).^2+(Kz-KzFilt).^2));

%Filter the images in the Fourier domain, and take the inverse transform
for j = 1:20
    UtFilt(j,:,:,:)=filter.*(squeeze(Utinstance(j,:,:,:)));
    UnFilt(j,:,:,:)=ifftn(UtFilt(j,:,:,:));
end


%Calculate a position for each image and plot its motion through the image
%sequence
GraphData = zeros(3,20)
for j = 1:20
    UnFiltInstance = abs(squeeze(UnFilt(j,:,:,:)));
    [C,In]=max(UnFiltInstance(:));
    [In1,In2,In3] = ind2sub(size(UnFiltInstance),In);
    xLoc = x(In1);
    yLoc = y(In2);
    zLoc = z(In3);
    GraphData(1,j)=xLoc;
    GraphData(2,j)=yLoc;
    GraphData(3,j)=zLoc;
end

%UnFiltSlice = squeeze(UnFiltInstance(:,:,In3));
%pcolor(x,y,UnFiltSlice);
plot3(GraphData(1,:),GraphData(2,:),GraphData(3,:));

    
