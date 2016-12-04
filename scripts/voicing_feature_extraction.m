% directorio de archivos
addpath ../audio
addpath ../funciones

clear variables; close all;

%% LOAD FILE
filename = 'ulla_mono.wav';
emb_gt = csvread('ulla_embrochure.csv');
[x, fs] = audioread(filename);
t=(0:1:length(x)-1)*(1/fs);

winlen = 1024;
hop = 512;
overlap = winlen - hop;
num_lags = 250;

% [s,f,t_s] = spectrogram(x,hamming(winlen),overlap,winlen,fs);

% figure();
% ax1(1) = subplot(4,1,4); plot(t,x,'-r'); 
%     title('Forma de onda'); xlabel('Tiempo (s)');
%     ylabel('x'); axis([0 max(t) -1 1]);
% ax1(2) = subplot(4,1,1:3); imagesc(t_s, f, 20*log(abs(s))), 
%     axis xy;
%     title('Representacion Tiempo-Frecuencia'); %xlabel('Tiempo (s)'); 
%     ylabel('Frecuencia (Hz)'); % axis([0 max(t_e) 0 1]);
% linkaxes(ax1,'x');

%% Features

zcr_thr = 0.5;
[zcr_tc, zcr_tc_norm, t_zcr, x_thr_zcr] = zero_crossing_rate(x, winlen+num_lags, hop, fs, zcr_thr);

ste_thr = 0.01;
[ste_tc, ste_tc_norm, t_ste, x_thr_ste] = short_time_energy(x, winlen+num_lags, hop, fs, ste_thr);

kappa = 0.2;
[voicing, t_voicing] = voicing(x, winlen, hop, fs, num_lags, kappa);

gt = zeros(size(t_voicing));

for i=1:length(emb_gt)-1
   gt( (t_voicing>=emb_gt(i,1))&(t_voicing<emb_gt(i+1,1)) )= emb_gt(i,2);
end

% csvwrite('zero_crossing_rate.csv',[t_zcr; zcr_tc_norm]');
% csvwrite('short_time_energy.csv',[t_ste; ste_tc_norm]');
csvwrite('ulla_voicing_matlab.csv',[voicing; t_voicing]');

% figure('Name',['Forma de onda y features de tiempo corto ' filename]);
% ax3(1) = subplot(3,1,1); plot(t,x,'-r');
%     hold on, hold off;
%     title('Forma de onda'); %xlabel('Tiempo (s)');
%     ylabel('x'); axis([0 max(t) -1 1]);
% ax3(2) = subplot(3,1,2:3);
%     hold on, plot(t_zcr,zcr_tc_norm), plot(t_voicing,voicing) 
%     plot(t_ste, ste_tc_norm);
%     hold off; title('Energia y Tasa de Cruces por Cero (normalizadas)'); 
%     xlabel('Tiempo (s)'); ylabel('STE_n / SCR_n'); axis([0 max(t_zcr) 0 1]);
%     grid on, legend('ZCR', 'voicing', 'STE')
% linkaxes(ax3,'x');

%% SAVE .MAT
% save('ulla_voicing_features.mat', 'gt', 'voicing', 'zcr_tc_norm', 'ste_tc_norm', 't_voicing')