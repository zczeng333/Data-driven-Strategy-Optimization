clc;
clear;
close all;

%% 初始化变量
global TimePointer;% 时间指针
TimePointer=1;
global Accumulation;% 储热清零
Accumulation=0;
global strategy;%记录优化后的策略及reward
strategy=[];
LoadData;% 载入数据

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
%%
%创建DQN
statePath = [
    imageInputLayer([2 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(24,'Name','CriticStateFC2')];
actionPath = [
    imageInputLayer([4 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(24,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','output')];
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
figure
plot(criticNetwork)
%%%
criticOpts = rlRepresentationOptions('LearnRate',0.0005,'GradientThreshold',1,'L2RegularizationFactor',0.001);
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'state'},'Action',{'action'},criticOpts);
agentOpts = rlDQNAgentOptions('UseDoubleDQN',false, ...% 是否使用double DQN
    'ExperienceBufferLength',100000, ...% 过往experience的长度
    'DiscountFactor',0.99, ...% MDP的折扣因子
    'MiniBatchSize',100);% 每个Episode从过往experience中采样的次数（值越大计算越慢，但方差越小，收敛越快）
agent = rlDQNAgent(critic,agentOpts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 860, ...% 没有meet终止条件时的最大episode数
    'MaxStepsPerEpisode', 1000, ...% 没有meet reset条件时每个episode的最大step数
    'SaveAgentCriteria',"EpisodeReward",...% 保存agent的判据
    'SaveAgentValue',50000,...% 对应判据的数值
    'SaveAgentDirectory', "Agents",...% 保存路径
    'Verbose', false, ...% 是否在命令行展示训练过程
    'Plots','training-progress',...% 图形化展示训练过程
    'StopTrainingCriteria','AverageReward',...% 停止训练的判据
    'StopTrainingValue',200000); % 对应判据的数值
doTraining = true;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
    xlswrite('Strategy.xlsx',strategy);
else
    % Load pretrained agent for the example.
    load('Agents/finalAgent.mat','agent');
    simOptions = rlSimulationOptions('MaxSteps',941);
    experience = sim(env,agent,simOptions);
end