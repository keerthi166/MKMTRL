function [a, cost, err, reg] = kernelRegression(K,y,gamma,useCG)

n = size(K,1);
if ~exist('useCG','var')
    useCG=false;
end
if nargin==3 || ~useCG
    a = (K+gamma*eye(n))\y;
else
    [a,~,~,~,~]=pcg(K+gamma*eye(n),y);
end
cost = gamma * y' * a;
err = gamma^2 * (a'*a);
reg = cost - err;


end
