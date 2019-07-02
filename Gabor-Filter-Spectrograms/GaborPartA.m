clear all; close all; clc

windowWidth = 5000;
translationStep = 0.1;

tr_piano=16; % record time in seconds
y=audioread('music1.wav'); 
y=decimate(y,4); %Reduce sample rate to save memory
Fs=length(y)/tr_piano;

L=length(y)/Fs; n=length(y);
t2=linspace(0,L,n+1); t=t2(1:n);
k=(2*pi/L)*[0:n/2-1 -n/2:-1]; ks=fftshift(k);

kmax = n/(40*L);

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
%title('Piano Spectrogram')

notes = [9,14,19,24,29,34,39,47,52,57,66,71,75,85,88,94,98,102,107,112,116,121,126,130,135,140];
frequencyIndices = [];

for j = 1:26
    [C,I]= max(Vgt_spec(notes(j),length(y)/2:length(y)));
    frequencyIndices = [frequencyIndices, I+length(y)/2];
end

%example of a spectral slice
%plot(ks(87680:175360)/(4*pi),Vgt_spec(9,87680:175360)/max(Vgt_spec(9,87680:175360)))
%xlabel('frequency (Hz')
%ylabel('amplitude (scaled to max)')
%title('Piano Spectral Slice')

%xaxis = linspace(1,26,26);
%scatter(xaxis,ks(frequencyIndices)/(4*pi))
%xlabel('notes in order')
%ylabel('frequency (Hz)')
%title('Piano Notes')

