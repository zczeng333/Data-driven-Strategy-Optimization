function [rm,rv,hm,hr]=cal_para(matrix)
%% �����������matrix������Ӧ��reward��heat�ľ�ֵ���׼��
rm=mean(matrix(:,6)/60);
rv=sqrt(var(matrix(:,6)/60));
hm=mean(matrix(:,7));
hr=sqrt(var(matrix(:,7)));
end