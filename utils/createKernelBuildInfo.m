function [ kernelInfo ] = createKernelBuildInfo(kernelType,kernelOpts,variableType,dim)
%CREATEKERNELBUILDINFO Create a structure that contains the information to
%build the kernel
%
% Sample Input Param
% kernelType={'gaussian' 'gaussian' 'poly' 'poly' };
% kernelOpts={[0.5 1 2 5 7 10 12 15 17 20] [0.5 1 2 5 7 10 12 15 17 20] [1 2 3] [1 2 3]};
% variableType={'all' 'single' 'all' 'single'};
%
% See also constructKernel.m


kt=1;
kernelInfo=[];
for i=1:length(kernelType);
    for k=1:length(kernelOpts{i})
        switch variableType{i}
            case 'all'
                kernelInfo(kt).kernel=kernelType{i};
                kernelInfo(kt).kernelOpts=kernelOpts{i}(k);
                kernelInfo(kt).variables=1:dim;
                kt=kt+1;
            case 'single'
                for j=1:dim
                    kernelInfo(kt).kernel=kernelType{i};
                    kernelInfo(kt).kernelOpts=kernelOpts{i}(k);
                    kernelInfo(kt).variables=j;
                    kt=kt+1;
                end
            case 'random'
                kernelInfo(kt).kernel=kernelType{i};
                kernelInfo(kt).kernelOpts=kernelOpts{i}(k);
                randIds=randperm(dim);
                nRd=floor(rand*dim)+1;
                kernelInfo(kt).variables=randIds(1:nRd);
                kt=kt+1;
        end
    end
end

end


