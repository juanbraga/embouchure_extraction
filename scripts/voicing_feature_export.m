clear all, close all

emma_features = load('emma_voicing_features.mat');
emma_voicing=emma_features.voicing;
emma_zcr=emma_features.zcr_tc_norm;
emma_gt=emma_features.gt;

pablo_features = load('pablo_voicing_features.mat');
pablo_voicing=pablo_features.voicing;
pablo_zcr=pablo_features.zcr_tc_norm;
pablo_gt=pablo_features.gt;

ulla_features = load('ulla_voicing_features.mat');
ulla_voicing=ulla_features.voicing;
ulla_zcr=ulla_features.zcr_tc_norm;
ulla_gt=ulla_features.gt;

% SCATTER PLOT
figure, subplot(3,3,1)
title('BHC Vs. Breathy: Emma Resmini'), hold on 
scatter(emma_voicing(emma_gt==1), emma_zcr(emma_gt==1)); hold on;
scatter(emma_voicing(emma_gt==2), emma_zcr(emma_gt==2)); hold on;
% scatter(emma_voicing(emma_gt==3), emma_zcr(emma_gt==3)); hold on;
% legend('Blow Hole Covert', 'Breathy', 'Normal Embrouchre')
xlabel('voicing'), ylabel('zcr'), grid on, hold off;

subplot(3,3,2), title('BHC Vs. Breathy: Pablo Somma'), hold on 
scatter(pablo_voicing(pablo_gt==1), pablo_zcr(pablo_gt==1)); hold on;
scatter(pablo_voicing(pablo_gt==2), pablo_zcr(pablo_gt==2)); hold on;
% scatter(pablo_voicing(pablo_gt==3), pablo_zcr(pablo_gt==3)); hold on;
% legend('Blow Hole Covert', 'Breathy', 'Normal Embrouchre'),
xlabel('voicing'), ylabel('zcr'), grid on, hold off;

subplot(3,3,3), title('BHC Vs. Breathy: Ulla Suokko'), hold on 
scatter(ulla_voicing(ulla_gt==1), ulla_zcr(ulla_gt==1)); hold on;
scatter(ulla_voicing(ulla_gt==2), ulla_zcr(ulla_gt==2)); hold on;
% scatter(ulla_voicing(ulla_gt==3), ulla_zcr(ulla_gt==3)); hold on;
% legend('Blow Hole Covert', 'Breathy', 'Normal Embrouchre'),
xlabel('voicing'), ylabel('zcr'), grid on, hold off;

subplot(3,3,4)
title('BHC Vs. Normal: Emma Resmini'), hold on 
scatter(emma_voicing(emma_gt==1), emma_zcr(emma_gt==1)); hold on;
% scatter(emma_voicing(emma_gt==2), emma_zcr(emma_gt==2)); hold on;
scatter(emma_voicing(emma_gt==3), emma_zcr(emma_gt==3)); hold on;
% legend('Blow Hole Covert', 'Breathy', 'Normal Embrouchre')
xlabel('voicing'), ylabel('zcr'), grid on, hold off;

subplot(3,3,5), title('BHC Vs. Normal: Pablo Somma'), hold on 
scatter(pablo_voicing(pablo_gt==1), pablo_zcr(pablo_gt==1)); hold on;
% scatter(pablo_voicing(pablo_gt==2), pablo_zcr(pablo_gt==2)); hold on;
scatter(pablo_voicing(pablo_gt==3), pablo_zcr(pablo_gt==3)); hold on;
% legend('Blow Hole Covert', 'Breathy', 'Normal Embrouchre'),
xlabel('voicing'), ylabel('zcr'), grid on, hold off;

subplot(3,3,6), title('BHC Vs. Normal: Ulla Suokko'), hold on 
scatter(ulla_voicing(ulla_gt==1), ulla_zcr(ulla_gt==1)); hold on;
% scatter(ulla_voicing(ulla_gt==2), ulla_zcr(ulla_gt==2)); hold on;
scatter(ulla_voicing(ulla_gt==3), ulla_zcr(ulla_gt==3)); hold on;
% legend('Blow Hole Covert', 'Breathy', 'Normal Embrouchre'),
xlabel('voicing'), ylabel('zcr'), grid on, hold off;

subplot(3,3,7)
title('Breathy Vs. Normal: Emma Resmini'), hold on 
% scatter(emma_voicing(emma_gt==1), emma_zcr(emma_gt==1)); hold on;
scatter(emma_voicing(emma_gt==2), emma_zcr(emma_gt==2)); hold on;
scatter(emma_voicing(emma_gt==3), emma_zcr(emma_gt==3)); hold on;
% legend('Blow Hole Covert', 'Breathy', 'Normal Embrouchre')
xlabel('voicing'), ylabel('zcr'), grid on, hold off;

subplot(3,3,8), title('Breathy Vs. Normal: Pablo Somma'), hold on 
% scatter(pablo_voicing(pablo_gt==1), pablo_zcr(pablo_gt==1)); hold on;
scatter(pablo_voicing(pablo_gt==2), pablo_zcr(pablo_gt==2)); hold on;
scatter(pablo_voicing(pablo_gt==3), pablo_zcr(pablo_gt==3)); hold on;
% legend('Blow Hole Covert', 'Breathy', 'Normal Embrouchre'),
xlabel('voicing'), ylabel('zcr'), grid on, hold off;

subplot(3,3,9), title('Breathy Vs. Normal: para Ulla Suokko'), hold on 
% scatter(ulla_voicing(ulla_gt==1), ulla_zcr(ulla_gt==1)); hold on;
scatter(ulla_voicing(ulla_gt==2), ulla_zcr(ulla_gt==2)); hold on;
scatter(ulla_voicing(ulla_gt==3), ulla_zcr(ulla_gt==3)); hold on;
% legend('Blow Hole Covert', 'Breathy', 'Normal Embrouchre'),
xlabel('voicing'), ylabel('zcr'), grid on, hold off;

%% Save .csv for Weka
voicing = [ulla_voicing(ulla_gt==1|ulla_gt==2|ulla_gt==3) emma_voicing(emma_gt==1|emma_gt==2|emma_gt==3) pablo_voicing(pablo_gt==1|pablo_gt==2|pablo_gt==3)];
zcr = [ulla_zcr(ulla_gt==1|ulla_gt==2|ulla_gt==3) emma_zcr(emma_gt==1|emma_gt==2|emma_gt==3) pablo_zcr(pablo_gt==1|pablo_gt==2|pablo_gt==3)];
gt = [ulla_gt(ulla_gt==1|ulla_gt==2|ulla_gt==3) emma_gt(emma_gt==1|emma_gt==2|emma_gt==3) pablo_gt(pablo_gt==1|pablo_gt==2|pablo_gt==3)];

csvwrite('voicing_features_256_128.csv', [voicing; zcr; gt]');

