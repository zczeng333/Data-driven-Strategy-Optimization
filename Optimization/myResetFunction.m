function [InitialObservation, LoggedSignal] = myResetFunction()
%% ������״̬���ã���������+�㴢�ȣ�
% LoggedSignals:����һϵ�е���Ϣ
% InitialObservation: ���س�ʼ״̬

W = 0;% ��ҹ
A = 0;% �㴢��Accumulation

% ��logged signals����ʽ�������û���״̬
LoggedSignal.State = [W;A];
InitialObservation = LoggedSignal.State;

end