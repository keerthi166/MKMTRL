function [ K,G] = constructGaussianKernels(simMat,bandwidths)
%CONSTRUCTGAUSSIANKERNELS 
%   Detailed explanation goes here
G = [];
    iter = 1;
    for i = 1 : size(simMat,3)
        for j = 1 : length(bandwidths)
            K(:, :, iter) = exp(-bandwidths(j)*simMat(:,:,i));
            iter = iter+1;
        end
        G = [G; length(bandwidths)];
    end

end

