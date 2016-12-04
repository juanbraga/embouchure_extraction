clear all, close all

emma_features = load('emma_mfcc_features.mat');
emma_mfcc=emma_features.mfcc;
emma_gt=emma_features.gt;

pablo_features = load('pablo_mfcc_features.mat');
pablo_mfcc=pablo_features.mfcc;
pablo_gt=pablo_features.gt;

ulla_features = load('ulla_mfcc_features.mat');
ulla_mfcc=ulla_features.mfcc;
ulla_gt=ulla_features.gt;

%% Save .csv for Weka
mfcc_train = [emma_mfcc(:,emma_gt==1|emma_gt==2|emma_gt==3) pablo_mfcc(:,pablo_gt==1|pablo_gt==2|pablo_gt==3)];
gt_train = [emma_gt(emma_gt==1|emma_gt==2|emma_gt==3) pablo_gt(pablo_gt==1|pablo_gt==2|pablo_gt==3)];
csvwrite('mfcc_40_ulla_train.csv', [mfcc_train; gt_train]'); 

mfcc_test = ulla_mfcc(:,ulla_gt==1|ulla_gt==2|ulla_gt==3);
gt_test = ulla_gt(ulla_gt==1|ulla_gt==2|ulla_gt==3);
csvwrite('mfcc_40_ulla_test.csv', [mfcc_test; gt_test]'); 

%% Save .csv for Weka
mfcc_train = [ulla_mfcc(:,ulla_gt==1|ulla_gt==2|ulla_gt==3) pablo_mfcc(:,pablo_gt==1|pablo_gt==2|pablo_gt==3)];
gt_train = [ulla_gt(ulla_gt==1|ulla_gt==2|ulla_gt==3) pablo_gt(pablo_gt==1|pablo_gt==2|pablo_gt==3)];
csvwrite('mfcc_40_emma_train.csv', [mfcc_train; gt_train]'); 

mfcc_test = emma_mfcc(:,emma_gt==1|emma_gt==2|emma_gt==3);
gt_test = emma_gt(emma_gt==1|emma_gt==2|emma_gt==3);
csvwrite('mfcc_40_emma_test.csv', [mfcc_test; gt_test]'); 

%% Save .csv for Weka
mfcc_train = [emma_mfcc(:,emma_gt==1|emma_gt==2|emma_gt==3) pablo_mfcc(:,pablo_gt==1|pablo_gt==2|pablo_gt==3)];
gt_train = [emma_gt(emma_gt==1|emma_gt==2|emma_gt==3) pablo_gt(pablo_gt==1|pablo_gt==2|pablo_gt==3)];
csvwrite('mfcc_40_ulla_train.csv', [mfcc_train; gt_train]'); 

mfcc_test = ulla_mfcc(:,ulla_gt==1|ulla_gt==2|ulla_gt==3);
gt_test = ulla_gt(ulla_gt==1|ulla_gt==2|ulla_gt==3);
csvwrite('mfcc_40_ulla_test.csv', [mfcc_test; gt_test]'); 