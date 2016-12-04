% directorio de archivos
addpath ../audio
addpath ../funciones

clear variables; close all;

%% Load File
filename = 'UllaSuokko_mono.wav';
emb_gt = csvread('ulla_embrochure.csv');
[x, fs] = audioread(filename);
t=(0:1:length(x)-1)*(1/fs);

winlen = 1024;
hop = 512;
overlap = winlen - hop;

[s,f,t_s] = spectrogram(x,hamming(winlen),overlap,winlen,fs);

figure();
ax1(1) = subplot(4,1,4); plot(t,x,'-r'); 
    title('Forma de onda'); xlabel('Tiempo (s)');
    ylabel('x'); axis([0 max(t) -1 1]);
ax1(2) = subplot(4,1,1:3); imagesc(t_s, f, 20*log(abs(s))), 
    axis xy;
    title('Representacion Tiempo-Frecuencia'); %xlabel('Tiempo (s)'); 
    ylabel('Frecuencia (Hz)'); % axis([0 max(t_e) 0 1]);
linkaxes(ax1,'x');

%% Features
t_len = floor((length(x)-winlen)/hop);
t_slices = ((0:1:t_len-1)*(hop/fs))+(winlen/(2*fs));

p=26; %numero de polos
slices=zeros(winlen,t_len);
ak=zeros(p, t_len);
e_rms=zeros(1, t_len);
e_rms_norm=zeros(1, t_len);

for i = 0:t_len-1
   slices(:,i+1)=x(i*hop+1:i*hop+winlen).*hamming(winlen);
   [ak(:,i+1), e_rms , e_rms_norm] = lpc_analysis(slices(:,i+1), p);
end

%% Ground Truth
gt = zeros(size(t_slices));
for i=1:length(emb_gt)-1
   gt( (t_slices>=emb_gt(i,1))&(t_slices<emb_gt(i+1,1)) )= emb_gt(i,2);
end

%% SAVE .MAT
save('emma_lpc_features.mat', 'ak', 'gt', 't_slices')