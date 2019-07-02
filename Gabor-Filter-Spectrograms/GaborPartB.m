clear all; close all; clc

windowWidth = 5000;
translationStep = 0.1;

tr_recorder=14; % record time in seconds
y=audioread('music2.wav'); 
y=decimate(y,4); %Reduce sample rate to save memory
Fs=length(y)/tr_recorder;

L=length(y)/Fs; n=length(y);
t2=linspace(0,L,n+1); t=t2(1:n);
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; ks=fftshift(k);

kmax = n/(10*L);

v = y'/2;

Vgt_spec=[];
tslide=0:translationStep:L;
for j=1:length(tslide)
    g=exp(-1*windowWidth*(t-tslide(j)).^2); % Gabor
    Vg=g.*v; Vgt=fft(Vg);
    Vgt_spec=[Vgt_spec; abs(fftshift(Vgt))];
    %subplot(3,1,1), plot(t,vsub,'k',t,g,'r')
    %subplot(3,1,2), plot(t,Vg,'k')
    %subplot(3,1,3), plot(ks,abs(fftshift(Vgt))/max(abs(Vgt)))
    %axis([-50 50 0 1])
    %drawnow
    %pause(0.1)
end

%pcolor(tslide,ks/(4*pi),Vgt_spec.'), shading interp
%set(gca,'Ylim',[0 kmax],'Fontsize',[14])
%colormap(hot)
%xlabel('time (s)')
%ylabel('frequency (Hz)')
%title('Recorder Spectrogram')

notes = [3,8,11,17,21,26,30,40,43,49,57,61,65,75,79,84,89,93,97,102,107,112,116,121,125,130];
frequencyIndices = [];

for j = 1:26
    [C,I]= max(Vgt_spec(notes(j),length(y)/2:length(y)));
    frequencyIndices = [frequencyIndices, I+length(y)/2];
end

%example of a spectral slice
%plot(ks(length(y)/2:length(y))/(4*pi),Vgt_spec(21,length(y)/2:length(y))/max(Vgt_spec(21),length(y)/2:length(y)))
%xlabel('frequency (Hz)')
%ylabel('amplitude (scaled to max)')
%title('Recorder Spectral Slice')

xaxis = linspace(1,26,26);
scatter(xaxis,ks(frequencyIndices)/(4*pi))
xlabel('notes in order')
ylabel('frequency (Hz)')
title('Recorder Notes')

