function frames=input_load()
global regPATH;
%% set path
addpath('registration\');
addpath('data sets\');
regPATH='results\';% set saving path
if ~exist(regPATH,'dir')
    mkdir(regPATH);
end
%% load dataset and obtain mean figure  
load ('middle fonts.mat','frames');%load dataset
Meanframe=mean(frames,3);
end
