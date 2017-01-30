function [ Kcomb ] = weightSumKernels( K,weights )
%WEIGHTSUMKERNELS Compute the weighted sum of the kernels using weights
%variable
% @param K 3D matrix of multiple kernels
% @param weights weights givens for multiple kernels


nKernels=length(weights);
Kcomb=zeros(size(K(:,:,1)));
ind=find(weights);
for k=1:length(ind)
    Kcomb=Kcomb+K(:,:,ind(k))*weights(ind(k));
end


end

