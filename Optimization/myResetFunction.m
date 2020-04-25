function [InitialObservation, LoggedSignal] = myResetFunction()
%% 将环境状态重置（任意天气+零储热）
% LoggedSignals:保存一系列的信息
% InitialObservation: 返回初始状态

W = 0;% 黑夜
A = 0;% 零储热Accumulation

% 以logged signals的形式返回重置环境状态
LoggedSignal.State = [W;A];
InitialObservation = LoggedSignal.State;

end