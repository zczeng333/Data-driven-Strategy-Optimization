clc;
clear;
close all;

%% ��ʼ������
global TimePointer;% ʱ��ָ��
TimePointer=1;
global Accumulation;% ��������
Accumulation=0;
global strategy;%��¼�Ż���Ĳ��Լ�reward
strategy=[];
LoadData;% ��������

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
%%
%����DQN
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
agentOpts = rlDQNAgentOptions('UseDoubleDQN',false, ...% �Ƿ�ʹ��double DQN
    'ExperienceBufferLength',100000, ...% ����experience�ĳ���
    'DiscountFactor',0.99, ...% MDP���ۿ�����
    'MiniBatchSize',100);% ÿ��Episode�ӹ���experience�в����Ĵ�����ֵԽ�����Խ����������ԽС������Խ�죩
agent = rlDQNAgent(critic,agentOpts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 860, ...% û��meet��ֹ����ʱ�����episode��
    'MaxStepsPerEpisode', 1000, ...% û��meet reset����ʱÿ��episode�����step��
    'SaveAgentCriteria',"EpisodeReward",...% ����agent���о�
    'SaveAgentValue',50000,...% ��Ӧ�оݵ���ֵ
    'SaveAgentDirectory', "Agents",...% ����·��
    'Verbose', false, ...% �Ƿ���������չʾѵ������
    'Plots','training-progress',...% ͼ�λ�չʾѵ������
    'StopTrainingCriteria','AverageReward',...% ֹͣѵ�����о�
    'StopTrainingValue',200000); % ��Ӧ�оݵ���ֵ
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