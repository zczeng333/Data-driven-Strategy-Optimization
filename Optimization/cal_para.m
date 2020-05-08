function [rm,rv,hm,hr]=cal_para(matrix)
%% 根据输入矩阵matrix计算相应的reward及heat的均值与标准差
rm=mean(matrix(:,6)/60);
rv=sqrt(var(matrix(:,6)/60));
hm=mean(matrix(:,7));
hr=sqrt(var(matrix(:,7)));
end