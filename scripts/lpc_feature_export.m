clear all, close all

emma_features = load('emma_lpc_features.mat');
emma_lpc=emma_features.ak;
emma_gt=emma_features.gt;

pablo_features = load('pablo_lpc_features.mat');
pablo_lpc=pablo_features.ak;
pablo_gt=pablo_features.gt;

ulla_features = load('ulla_lpc_features.mat');
ulla_lpc=ulla_features.ak;
ulla_gt=ulla_features.gt;

%% Save .csv for Weka
lpc = [ulla_lpc(:,ulla_gt==1|ulla_gt==2|ulla_gt==3) emma_lpc(:,emma_gt==1|emma_gt==2|emma_gt==3) pablo_lpc(:,pablo_gt==1|pablo_gt==2|pablo_gt==3)];
gt = [ulla_gt(ulla_gt==1|ulla_gt==2|ulla_gt==3) emma_gt(emma_gt==1|emma_gt==2|emma_gt==3) pablo_gt(pablo_gt==1|pablo_gt==2|pablo_gt==3)];

csvwrite('lpc_features.csv', [lpc; gt]');

%% Save .csv for Weka
lpc_train = [emma_lpc(:,emma_gt==1|emma_gt==2|emma_gt==3) pablo_lpc(:,pablo_gt==1|pablo_gt==2|pablo_gt==3)];
gt_train = [emma_gt(emma_gt==1|emma_gt==2|emma_gt==3) pablo_gt(pablo_gt==1|pablo_gt==2|pablo_gt==3)];
csvwrite('lpc_26_ulla_train.csv', [lpc_train; gt_train]'); 

lpc_test = ulla_lpc(:,ulla_gt==1|ulla_gt==2|ulla_gt==3);
gt_test = ulla_gt(ulla_gt==1|ulla_gt==2|ulla_gt==3);
csvwrite('lpc_26_ulla_test.csv', [lpc_test; gt_test]');