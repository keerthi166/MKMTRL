function [M] = constructMTMKLKernel(X,kernelInfo,opts,Xtest,beta)
% MTMKLKERNEL Construct Multi-task Multiple kernels
% X - cell array of length T (1xT), Multi-task Data
%
% See also constructMKLKernel.m, createKernelBuildInfo.m


T=length(X);
M=cell(1,T); % Each cell element is  Nt x Nt x nKernels 3D matrix

for t=1:T
    if nargin < 4
        M{t}=constructMKLKernel(X{t},kernelInfo,opts);
    else
        if iscell(Xtest)
           Xt=Xtest{t};
        else
            Xt=Xtest;
        end
        M{t}=constructMKLKernel(X{t},kernelInfo,opts,Xt,beta{t});
    end
end
end

