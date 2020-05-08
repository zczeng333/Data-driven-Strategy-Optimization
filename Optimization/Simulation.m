clc;
clear;
close all;

LoadData;
state=xlsread('Original Strategy.csv');
len=size(state);
global TimePointer;% 时间指针
TimePointer=1;
global Accumulation;% 储热清零
Accumulation=0;
global strategy;% 记录优化后的策略及reward
strategy=[];
record=[];

%%
%观测空间
%两个观测量（天气状况（离散）->0/1/2；蓄热量（液位高度，连续））
ObservationInfo = rlNumericSpec([2 1]);
ObservationInfo.Name = 'Plant States';
ObservationInfo.Description = 'weather, accumulation';
%%
%动作空间
ActionInfo = rlFiniteSetSpec({...
    [0;0;0;0],[0;0;0;1],[1;0;0;1],[1;0;0;2],[1;1;0;0],...
    [1;1;0;1],[1;1;0;2],[1;1;1;0],[1;1;1;1],[1;1;1;2]});
ActionInfo.Name = 'Plant Actions';
ActionInfo.Description = 'light, absorption, storage, generation';
%%
%创建环境
env = rlFunctionEnv(ObservationInfo,ActionInfo,'myStepFunction','myResetFunction');

%% Simulation
rng(0);
InitialObs = reset(env);
for i=1:len(1)
    Action=state(i,2:5);
    Accumulation=100;
    [NextObs,Reward,IsDone,LoggedSignals] = step(env,Action');
    record=[record;Reward];
end
xlswrite('Record.xlsx',record);