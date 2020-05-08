%% 绘制策略图像
D=xlsread('Example.xlsx');
figure(1);
subplot(4,1,1);
plot(D(:,1),D(:,3),'k','LineWidth',1.5);
title('Concentrator');
xlabel('Time');
ylabel('Action');
hold on;
subplot(4,1,2);
plot(D(:,1),D(:,4),'k','LineWidth',1.5);
title('Absorbtor');
xlabel('Time');
ylabel('Action');
hold on;
subplot(4,1,3);
plot(D(:,1),D(:,5),'k','LineWidth',1.5);
title('Accumulator');
xlabel('Time');
ylabel('Action');
hold on;
subplot(4,1,4);
plot(D(:,1),D(:,6),'k','LineWidth',1.5);
title('Geneator');
xlabel('Time');
ylabel('Action');

figure(2);
subplot(4,1,1);
plot(D(:,1),D(:,8),'k','LineWidth',1.5);
title('Concentrator');
xlabel('Time');
ylabel('Action');
hold on;
subplot(4,1,2);
plot(D(:,1),D(:,9),'k','LineWidth',1.5);
title('Absorbtor');
xlabel('Time');
ylabel('Action');
hold on;
subplot(4,1,3);
plot(D(:,1),D(:,10),'k','LineWidth',1.5);
title('Accumulator');
xlabel('Time');
ylabel('Action');
hold on;
subplot(4,1,4);
plot(D(:,1),D(:,11),'k','LineWidth',1.5);
title('Geneator');
xlabel('Time');
ylabel('Action');

% 
% %% 绘制reward图像
% figure(2);
% subplot(2,1,1);
% plot(D(:,1),D(:,7));
% title('Original Reward');
% subplot(2,1,2);
% plot(D(:,1),D(:,8));
% title('Optimized Reward');

% %% 绘制发电量柱状图
% data=xlsread('Power.csv');
% x=[];
% for i=1:43 % 横坐标
%     x=[x,i];
% end
% figure;
% bar(x,data)
% % for i = 1:length(x)
% %     text(x(i)-0.2,data(i,1)),...
% %     'HorizontalAlignment','center',...
% %     'VerticalAlignment','bottom')
% %     text(x(i)+0.2,data(i,2)),...
% %     'HorizontalAlignment','center',...
% %     'VerticalAlignment','bottom')
% % end
% title('Comparason between Original and Optimized Strategy');
% xlabel('Batch');
% ylabel('Electricity');
% legend('Original Strategy','Optimized Strategy');
% % axis([0 5 0.0 50]);