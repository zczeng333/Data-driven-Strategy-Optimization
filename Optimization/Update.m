function [reward, heat] = Update(Action,State)
W=State(1);% 天气状态
A=State(2);% 储能状态

%% 根据选择的action，返回对应action的reward及储热增量
global M00000;
global M00001;
global M10000;
global M10001;
global M11101;
global M11102;
global M11110;
global M11111;
global M11112;
global M20000;
global M21101;
global M21102;
global M21110;
global M21111;
global M21112;

reward=0; % 不存在的动作，reward=0
heat=0;
if (W==0&&Action(1)~=0)||(Action(1)==0&&sum(Action(2:3))~=0) % 无光照时不能聚光、吸热、蓄热
    reward=-100;
elseif all(Action==[0 0 0 0])
    if W==0
        row=unidrnd(size(M00000,1));
        reward=M00000(row,6)/60;
        heat=M00000(row,7);
    elseif W==1
        row=unidrnd(size(M10000,1));
        reward=M10000(row,6)/60;
        heat=M10000(row,7);
    else
        row=unidrnd(size(M20000,1));
        reward=M20000(row,6)/60;
        heat=M20000(row,7);
    end
elseif all(Action==[0 0 0 1])
    if W==0
        row=unidrnd(size(M00001,1));
        reward=M00001(row,6)/60;
        heat=M00001(row,7);
    elseif W==1
        row=unidrnd(size(M10001,1));
        reward=M10001(row,6)/60;
        heat=M10001(row,7);
    end
elseif all(Action==[1 1 0 1])
    if W==1
        row=unidrnd(size(M11101,1));
        reward=M11101(row,6)/60;
        heat=M11101(row,7);
    elseif W==2
        row=unidrnd(size(M21101,1));
        reward=M21101(row,6)/60;
        heat=M21101(row,7);
    end
elseif all(Action==[1 1 0 2])
    if W==1
        row=unidrnd(size(M11102,1));
        reward=M11102(row,6)/60;
        heat=M11102(row,7);
    elseif W==2
        row=unidrnd(size(M21102,1));
        reward=M21102(row,6)/60;
        heat=M21102(row,7);
    end
elseif all(Action==[1 1 1 0])
    if W==1
        row=unidrnd(size(M11110,1));
        reward=M11110(row,6)/60;
        heat=M11110(row,7);
    elseif W==2
        row=unidrnd(size(M21110,1));
        reward=M21110(row,6)/60;
        heat=M21110(row,7);
    end
elseif all(Action==[1 1 1 1])
    if W==1
        row=unidrnd(size(M11111,1));
        reward=M11111(row,6)/60;
        heat=M11111(row,7);
    elseif W==2
        row=unidrnd(size(M21111,1));
        reward=M21111(row,6)/60;
        heat=M21111(row,7);
    end
elseif all(Action==[1 1 1 2])
    if W==1
        row=unidrnd(size(M11112,1));
        reward=M11112(row,6)/60;
        heat=M11112(row,7);
    elseif W==2
        row=unidrnd(size(M21112,1));
        reward=M21112(row,6)/60;
        heat=M21112(row,7);
    end
end
end