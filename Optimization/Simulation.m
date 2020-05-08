clc;
clear;
close all;

LoadData;
state=xlsread('Original Strategy.csv');
len=size(state);
global TimePointer;% ʱ��ָ��
TimePointer=1;
global Accumulation;% ��������
Accumulation=0;
global strategy;% ��¼�Ż���Ĳ��Լ�reward
strategy=[];
record=[];

%%
%�۲�ռ�
%�����۲���������״������ɢ��->0/1/2����������Һλ�߶ȣ���������
ObservationInfo = rlNumericSpec([2 1]);
ObservationInfo.Name = 'Plant States';
ObservationInfo.Description = 'weather, accumulation';
%%
%�����ռ�
ActionInfo = rlFiniteSetSpec({...
    [0;0;0;0],[0;0;0;1],[1;0;0;1],[1;0;0;2],[1;1;0;0],...
    [1;1;0;1],[1;1;0;2],[1;1;1;0],[1;1;1;1],[1;1;1;2]});
ActionInfo.Name = 'Plant Actions';
ActionInfo.Description = 'light, absorption, storage, generation';
%%
%��������
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