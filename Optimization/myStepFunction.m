function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunction(Action,LoggedSignals)
%% ����ѡ��actionӰ�컷���������ж�Ӧ��ϵͳ�仯
% Action: ����ѡ��Ķ���
% Weather: ��һʱ�̵�����
% LoggedSignals: ������һϵ���ź�ֵ
% NextObs: ���ݴ�action�õ��Ļ�����һ���۲�ֵ
% Reward: ���ݴ�action��õĻر�
% IsDone: �Ƿ����

global TimePointer;
global Weather;
global Accumulation;
global strategy;

% ��LoggedSignals����ȡ��ϵͳ״̬
State = LoggedSignals.State;

% ��ȡ��һʱ�̵�״̬
[Reward, heat]=Update(Action',State);% ��ȡ��Ӧaction��reward�����ܱ仯
TimePointer=TimePointer+1;
if Accumulation<=0&&heat<0 % �����Ȳ��������ȵ����
    Reward=min(Reward,-10);
%     Accumulation=0;
% else % ���������
%     Accumulation=Accumulation+heat;
end
Accumulation=Accumulation+heat;
strategy=[strategy;[State(1),Action',Reward,Accumulation]];%��¼strategy
if TimePointer==size(Weather,1)% ����TimePointer
    TimePointer=1;
end
LoggedSignals.State = [Weather(TimePointer);Accumulation];

% ��״̬ת��Ϊ�۲�ֵ
NextObs = LoggedSignals.State;

%���batch��ֹ����
if mod(TimePointer,941)==1% ���һ��episode(һ��)
    IsDone=true;
    Accumulation=0;
else
    IsDone=false;
end
