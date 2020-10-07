x0 = [1 1];                        % Starting guess
%[x,resnorm] = lsqnonlin(@myfun,x0,[],[],optimset('Display','iter','MaxIter',100));
[x,resnorm] = lsqnonlin(@(x) sin(3*x),x0,[],[],optimset('Display','iter','MaxIter',100));