function [reward, heat] = Update(Action,State)
W=State(1);% 天气状态
A=State(2);% 储能状态

%% 根据选择的action，返回对应action的reward及储热增量
global M00000;
global M00001;
global M10000;
global M10001;
global M11000;
global M11001;
global M11002;
global M11100;
global M11101;
global M11110;
global M11111;
global M12001;
global M12002;
global M12100;
global M12101;
global M12102;
global M12110;
global M12111;
global M12112;
global M20000;
global M21000;
global M21001;
global M21002;
global M21100;
global M21101;
global M21102;
global M21110;
global M21111;
global M22001;
global M22002;
global M22101;
global M22102;
global M22110;
global M22111;
global M22112;
reward=0;
heat=0;
if A==0&&Action(1)==0&&Action(4)~=0 % 无储能且无聚光时不能发电
    reward=0;
    heat=0;
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
elseif all(Action==[1 0 0 0])
    if W==1
        row=unidrnd(size(M11000,1));
        reward=M11000(row,6)/60;
        heat=M11000(row,7);
    elseif W==2
        row=unidrnd(size(M21000,1));
        reward=M21000(row,6)/60;
        heat=M21000(row,7);
    end
elseif all(Action==[1 0 0 1])
    if W==1
        row=unidrnd(size(M11001,1));
        reward=M11001(row,6)/60;
        heat=M11001(row,7);
    elseif W==2
        row=unidrnd(size(M21001,1));
        reward=M21001(row,6)/60;
        heat=M21001(row,7);
    end
elseif all(Action==[1 0 0 2])
    if W==1
        row=unidrnd(size(M11002,1));
        reward=M11002(row,6)/60;
        heat=M11002(row,7);
    elseif W==2
        row=unidrnd(size(M21002,1));
        reward=M21002(row,6)/60;
        heat=M21002(row,7);
    end
elseif all(Action==[1 1 0 0])
    if W==1
        row=unidrnd(size(M11100,1));
        reward=M11100(row,6)/60;
        heat=M11100(row,7);
    elseif W==2
        row=unidrnd(size(M21100,1));
        reward=M21100(row,6)/60;
        heat=M21100(row,7);
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
    if W==2
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
elseif all(Action==[2 0 0 1])
    if W==1
        row=unidrnd(size(M12001,1));
        reward=M12001(row,6)/60;
        heat=M12001(row,7);
    elseif W==2
        row=unidrnd(size(M22001,1));
        reward=M22001(row,6)/60;
        heat=M22001(row,7);
    end
elseif all(Action==[2 0 0 2])
    if W==1
        row=unidrnd(size(M12002,1));
        reward=M12002(row,6)/60;
        heat=M12002(row,7);
    elseif W==2
        row=unidrnd(size(M22002,1));
        reward=M22002(row,6)/60;
        heat=M22002(row,7);
    end
elseif all(Action==[2 1 0 0])
    if W==1
        row=unidrnd(size(M12100,1));
        reward=M12100(row,6)/60;
        heat=M12100(row,7);
    end
elseif all(Action==[2 1 0 1])
    if W==1
        row=unidrnd(size(M12101,1));
        reward=M12101(row,6)/60;
        heat=M12101(row,7);
    elseif W==2
        row=unidrnd(size(M22101,1));
        reward=M22101(row,6)/60;
        heat=M22101(row,7);
    end
elseif all(Action==[2 1 0 2])
    if W==1
        row=unidrnd(size(M12102,1));
        reward=M12102(row,6)/60;
        heat=M12102(row,7);
    elseif W==2
        row=unidrnd(size(M22102,1));
        reward=M22102(row,6)/60;
        heat=M22102(row,7);
    end
elseif all(Action==[2 1 1 0])
    if W==1
        row=unidrnd(size(M12110,1));
        reward=M12110(row,6)/60;
        heat=M12110(row,7);
    elseif W==2
        row=unidrnd(size(M22110,1));
        reward=M22110(row,6)/60;
        heat=M22110(row,7);
    end
elseif all(Action==[2 1 1 1])
    if W==1
        row=unidrnd(size(M12111,1));
        reward=M12111(row,6)/60;
        heat=M12111(row,7);
    elseif W==2
        row=unidrnd(size(M22111,1));
        reward=M22111(row,6)/60;
        heat=M22111(row,7);
    end
elseif all(Action==[2 1 1 2])
    if W==1
        row=unidrnd(size(M12112,1));
        reward=M12112(row,6)/60;
        heat=M12112(row,7);
    elseif W==2
        row=unidrnd(size(M22112,1));
        reward=M22112(row,6)/60;
        heat=M22112(row,7);
    end
end
end