function K=constructMKLKernel(xapp,kernelInfo,opts,xsup,beta)
% Construct Multiple Kernels for a given data X of size NxP based on
% kernelInfo
% @param X input data of size NxP
% @param kernelInfo cell array of structs of length (#Kernels)
% beta is the support vector
% See also constructKernel.m,createKernelBuildInfo.m

nKernels=length(kernelInfo);
K=[];
if nargin <4
    xsup=xapp;
    for k=1:nKernels
        Kr=constructKernel(xapp(:,kernelInfo(k).variables),kernelInfo(k).kernel,kernelInfo(k).kernelOpts, xsup(:,kernelInfo(k).variables));
        if opts.efficientkernel
            Kr=build_efficientK(Kr);
        end
        
        K(:,:,k)=Kr;
    end
else
    % Construct Kernel matrix between xapp and xsup
    ind=find(beta);
    K=zeros(size(xapp,1),size(xsup,1));
    for i=1:length(ind);
        k=ind(i);
        Kr=constructKernel(xapp(:,kernelInfo(k).variables),kernelInfo(k).kernel,kernelInfo(k).kernelOpts, xsup(:,kernelInfo(k).variables));
        K=K+ Kr*beta(k);
    end
    
end

end