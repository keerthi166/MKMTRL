function [K,normK] = normalizeKernel(K,normK,kopts)
%NORMALIZEKERNEL Perform Unit Trace Normalization of the Kernel(s) K
% @param K can be 3D kernel for multiple kernel learning
% @normK unit trace normalization for the kernels (optional)

nKernels=size(K,3); % if nKernels>1 then Normalize Multiple Kernels
tol=1e-8;
correctEig=false;
if nargin==3
    if isfield(kopts,'correctEig')
        correctEig=kopts.correctEig;
    end
end
isComputeNorm=1;
if nargin >1
    if ~isempty(normK)
        isComputeNorm=0;
    end
end
for k=1:nKernels
    if isComputeNorm
        normK(k,1) = trace(K(:,:,k));
    end
    if normK(k)>tol
        K(:,:,k)=K(:,:,k)./normK(k);
    else
        normK(k)=1;
    end
    if correctEig
        minEps = min(eig(K(:,:,k)));
        if (minEps <= 0)
            K(:,:,k) = K(:,:,k) + (abs(minEps)+tol)*eye(size(K(:,:,1),1));
        end
    end
end


end

