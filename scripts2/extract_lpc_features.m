%% PROYECTO FIN DE CURSO AUDIODSP 2016 - IIE UDELAR
clear all;
close all;
% addpath ../funciones
% addpath ../audio

% function result = extract_lpc_features(p, winlen, hop)

addpath ../funciones
addpath ../audio
warning('off','all')

%% IMPORT FILES
claire_audio_file = '../audio/claire_mono.wav'; 
claire_gt_file = '../audio/claire_embrochure.csv';
juan_audio_file = '../audio/juan_mono.wav'; 
juan_gt_file = '../audio/juan_embrochure.csv';
emma_audio_file = '../audio/emma_mono.wav'; 
emma_gt_file = '../audio/emma_embrochure.csv';
pablo_audio_file = '../audio/pablo_mono.wav'; 
pablo_gt_file = '../audio/pablo_embrochure.csv';
ulla_audio_file = '../audio/ulla_mono.wav'; 
ulla_gt_file = '../audio/ulla_embrochure.csv';

[x_claire, fs] = audioread(claire_audio_file);
% t_claire=(0:1:length(x_claire)-1)*(1/fs);

[x_juan, fs] = audioread(juan_audio_file);
% t_juan=(0:1:length(x_juan)-1)*(1/fs);

[x_emma, fs] = audioread(emma_audio_file);
% t_emma=(0:1:length(x_emma)-1)*(1/fs);

[x_pablo, fs] = audioread(pablo_audio_file);
% t_pablo=(0:1:length(x_pablo)-1)*(1/fs);

[x_ulla, fs] = audioread(ulla_audio_file);
% t_ulla=(0:1:length(x_ulla)-1)*(1/fs);

%% FEATURE EXTRACTION
winlen = 1024;
hop = 512;
overlap = winlen - hop;

[slices_claire, t_slices_claire] = slice_audio(x_claire, fs, winlen, hop);
[slices_juan, t_slices_juan] = slice_audio(x_juan, fs, winlen, hop);
[slices_emma, t_slices_emma] = slice_audio(x_emma, fs, winlen, hop);
[slices_pablo, t_slices_pablo] = slice_audio(x_pablo, fs, winlen, hop);
[slices_ulla, t_slices_ulla] = slice_audio(x_ulla, fs, winlen, hop);

gt_claire = get_ground_truth(csvread(claire_gt_file), t_slices_claire);
gt_juan = get_ground_truth(csvread(juan_gt_file), t_slices_juan);
gt_emma = get_ground_truth(csvread(emma_gt_file), t_slices_emma);
gt_pablo = get_ground_truth(csvread(pablo_gt_file), t_slices_pablo);
gt_ulla = get_ground_truth(csvread(ulla_gt_file), t_slices_ulla);

t_len_claire = length(t_slices_juan);
t_len_juan = length(t_slices_juan);
t_len_emma = length(t_slices_emma);
t_len_pablo = length(t_slices_pablo);
t_len_ulla = length(t_slices_ulla);

p=20;

%CLAIRE
ak_claire = zeros(p, t_len_claire);
e_rms_claire = zeros(1, t_len_claire);
e_rms_norm_claire = zeros(1, t_len_claire);
for i = 0:t_len_claire-1
   [ak_claire(:,i+1), e_rms_claire , e_rms_norm_claire] = ... 
       lpc_analysis(slices_claire(:,i+1), p);  
end

%JUAN 
ak_juan = zeros(p, t_len_juan);
e_rms_juan = zeros(1, t_len_juan);
e_rms_norm_juan = zeros(1, t_len_juan);
for i = 0:t_len_juan-1
   [ak_juan(:,i+1), e_rms_juan , e_rms_norm_juan] = ... 
       lpc_analysis(slices_juan(:,i+1), p);  
end

%EMMA 
ak_emma = zeros(p, t_len_emma);
e_rms_emma = zeros(1, t_len_emma);
e_rms_norm_emma = zeros(1, t_len_emma);
for i = 0:t_len_emma-1
   [ak_emma(:,i+1), e_rms_emma , e_rms_norm_emma] = ... 
       lpc_analysis(slices_emma(:,i+1), p);  
end

%PABLO
ak_pablo = zeros(p, t_len_pablo);
e_rms_pablo = zeros(1, t_len_pablo);
e_rms_norm_pablo = zeros(1, t_len_pablo);
for i = 0:t_len_pablo-1
    [ak_pablo(:,i+1), e_rms_pablo , e_rms_norm_pablo] = ... 
        lpc_analysis(slices_pablo(:,i+1), p);
end

%ULLA
ak_ulla = zeros(p, t_len_ulla);
e_rms_ulla = zeros(1, t_len_ulla);
e_rms_norm_ulla = zeros(1, t_len_ulla);
for i = 0:t_len_ulla-1
    [ak_ulla(:,i+1), e_rms_ulla , e_rms_norm_ulla] = ... 
        lpc_analysis(slices_ulla(:,i+1), p);
end

% EXTRACT CLAIRE EMB
claire_train = [ak_emma(:,gt_emma==1|gt_emma==2|gt_emma==3) ...
    ak_pablo(:,gt_pablo==1|gt_pablo==2|gt_pablo==3) ...
    ak_ulla(:,gt_ulla==1|gt_ulla==2|gt_ulla==3)]';
claire_gt_train = [gt_emma(gt_emma==1|gt_emma==2|gt_emma==3) ...
    gt_pablo(gt_pablo==1|gt_pablo==2|gt_pablo==3) ...
    gt_ulla(gt_ulla==1|gt_ulla==2|gt_ulla==3)]';
claire_test = ak_claire(:,gt_claire==1|gt_claire==2|gt_claire==3)';
claire_gt_test = gt_claire(gt_claire==1|gt_claire==2|gt_claire==3)';

claire_train = [claire_train claire_gt_train]';
claire_test = [claire_test claire_gt_test]';
filename_aux=strcat('../features/claire_lpc_',int2str(p),'_train.mat');
save(filename_aux,'claire_train')
filename_aux=strcat('../features/claire_lpc_',int2str(p),'_test.mat');
save(filename_aux,'claire_test')

% EXTRACT ULLA EMB
ulla_train = [ak_emma(:,gt_emma==1|gt_emma==2|gt_emma==3) ...
    ak_pablo(:,gt_pablo==1|gt_pablo==2|gt_pablo==3)]';
ulla_gt_train = [gt_emma(gt_emma==1|gt_emma==2|gt_emma==3) ...
    gt_pablo(gt_pablo==1|gt_pablo==2|gt_pablo==3)]';
ulla_test = ak_ulla(:,gt_ulla==1|gt_ulla==2|gt_ulla==3)';
ulla_gt_test = gt_ulla(gt_ulla==1|gt_ulla==2|gt_ulla==3)';

ulla_train = [ulla_train ulla_gt_train]';
ulla_test = [ulla_test ulla_gt_test]';
filename_aux=strcat('../features/ulla_lpc_',int2str(p),'_train.mat');
save(filename_aux,'ulla_train')
filename_aux=strcat('../features/ulla_lpc_',int2str(p),'_test.mat');
save(filename_aux,'ulla_test')

% EXTRACT PABLO EMB
pablo_train = [ak_emma(:,gt_emma==1|gt_emma==2|gt_emma==3) ...
    ak_ulla(:,gt_ulla==1|gt_ulla==2|gt_ulla==3)]';
pablo_gt_train = [gt_emma(gt_emma==1|gt_emma==2|gt_emma==3) ...
    gt_ulla(gt_ulla==1|gt_ulla==2|gt_ulla==3)]';
pablo_test = ak_pablo(:,gt_pablo==1|gt_pablo==2|gt_pablo==3)';
pablo_gt_test = gt_pablo(gt_pablo==1|gt_pablo==2|gt_pablo==3)';

pablo_train = [pablo_train pablo_gt_train]';
pablo_test = [pablo_test pablo_gt_test]';
filename_aux=strcat('../features/pablo_lpc_',int2str(p),'_train.mat');
save(filename_aux,'pablo_train')
filename_aux=strcat('../features/pablo_lpc_',int2str(p),'_test.mat');
save(filename_aux,'pablo_test')

% EXTRACT EMMA EMB
emma_train = [ak_ulla(:,gt_ulla==1|gt_ulla==2|gt_ulla==3) ...
    ak_pablo(:,gt_pablo==1|gt_pablo==2|gt_pablo==3)]';
emma_gt_train = [gt_ulla(gt_ulla==1|gt_ulla==2|gt_ulla==3) ...
    gt_pablo(gt_pablo==1|gt_pablo==2|gt_pablo==3)]';
emma_test = ak_emma(:,gt_emma==1|gt_emma==2|gt_emma==3)';
emma_gt_test = gt_emma(gt_emma==1|gt_emma==2|gt_emma==3)';

emma_train = [emma_train emma_gt_train]';
emma_test = [emma_test emma_gt_test]';

filename_aux=strcat('../features/emma_lpc_',int2str(p),'_train.mat');
save(filename_aux,'emma_train')
filename_aux=strcat('../features/emma_lpc_',int2str(p),'_test.mat');
save(filename_aux,'emma_test')

% EXTRACT JUAN EMB
juan_train = [ak_ulla(:,gt_ulla==1|gt_ulla==2|gt_ulla==3) ...
    ak_pablo(:,gt_pablo==1|gt_pablo==2|gt_pablo==3) ... 
    ak_emma(:,gt_emma==1|gt_emma==2|gt_emma==3)]';
juan_gt_train = [gt_ulla(gt_ulla==1|gt_ulla==2|gt_ulla==3) ...
    gt_pablo(gt_pablo==1|gt_pablo==2|gt_pablo==3) ...
    gt_emma(gt_emma==1|gt_emma==2|gt_emma==3)]';
juan_test = ak_juan(:,gt_juan==1|gt_juan==2|gt_juan==3)';
juan_gt_test = gt_juan(gt_juan==1|gt_juan==2|gt_juan==3)';

juan_train = [juan_train juan_gt_train]';
juan_test = [juan_test juan_gt_test]';

filename_aux=strcat('../features/juan_lpc_',int2str(p),'_train.mat');
save(filename_aux,'juan_train')
filename_aux=strcat('../features/juan_lpc_',int2str(p),'_test.mat');
save(filename_aux,'juan_test')

% %% CONVERT FROM MAT TO NPY
% command=strcat('python mat2npy_lpc_features.py', ...
%     ' ../features/ulla_lpc_', int2str(p), ' ulla');
% dos(command)
% 
% command=strcat('python mat2npy_lpc_features.py', ...
%     ' ../features/pablo_lpc_', int2str(p), ' pablo');
% dos(command)
% 
% command=strcat('python mat2npy_lpc_features.py', ...
%     ' ../features/emma_lpc_', int2str(p), ' emma');
% dos(command)
% 
% command=strcat('python mat2npy_lpc_features.py', ...
%     ' ../features/claire_lpc_', int2str(p), ' claire');
% dos(command)
% 
% command=strcat('python mat2npy_lpc_features.py', ...
%     ' ../features/juan_lpc_', int2str(p), ' juan');
% dos(command)
% 
% result = 1; 
