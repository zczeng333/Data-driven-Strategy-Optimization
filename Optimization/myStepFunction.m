function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunction(Action,LoggedSignals)
%% 依据选择action影响环境，并评判对应的系统变化
% Action: 当下选择的动作
% Weather: 下一时刻的天气
% LoggedSignals: 过往的一系列信号值
% NextObs: 根据此action得到的环境下一个观测值
% Reward: 根据此action获得的回报
% IsDone: 是否完成

global TimePointer;
global Weather;
global Accumulation;
global strategy;

% 从LoggedSignals中提取出系统状态
State = LoggedSignals.State;

% 获取下一时刻的状态
[Reward, heat]=Update(Action',State);% 获取对应action的reward及储能变化
TimePointer=TimePointer+1;
if Accumulation<=0&&heat<0 % 负蓄热并持续耗热的情况
    Reward=min(Reward,-10);
%     Accumulation=0;
% else % 正蓄热情况
%     Accumulation=Accumulation+heat;
end
Accumulation=Accumulation+heat;
strategy=[strategy;[State(1),Action',Reward,Accumulation]];%记录strategy
if TimePointer==size(Weather,1)% 重置TimePointer
    TimePointer=1;
end
LoggedSignals.State = [Weather(TimePointer);Accumulation];

% 将状态转化为观测值
NextObs = LoggedSignals.State;

%检查batch终止条件
if mod(TimePointer,941)==1% 完成一个episode(一天)
    IsDone=true;
    Accumulation=0;
else
    IsDone=false;
end
