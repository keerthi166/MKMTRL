function [Xtrain,Xtest,meanX,stdX] = normalizeMultitaskData(Xtrain,Xtest,meanX,stdX)
% Normalize Multitask Dataset - mean and standard deviation to 0 and 1
% USAGE
%
%  [Xtrain,Xtest,meanX,stdX] = normalizeMultitaskData(Xtrain,Xtest)
% Xtrain = cell array of length 1xT
% Xtest = cell array of length 1xT
% meanX = cell array of length 1xT
% stdX = cell array of length 1xT

tol=1e-5;
T=length(Xtrain);
for t=1:T
    xapp=Xtrain{t};
    
    if nargin <3
        meanX{t}=mean(xapp);
        stdX{t}=std(xapp);
    end
    nbxapp=size(xapp,1);
    indzero=find(abs(stdX{t})<tol);
    if ~isempty(indzero)
        stdX{t}(indzero)=1;
    end;
    
    xapp= (xapp - ones(nbxapp,1)*meanX{t})./ (ones(nbxapp,1)*stdX{t}) ;
    Xtrain{t}=xapp;
    if nargin >1 && ~isempty(Xtest)
        xtest=Xtest{t};
        nbxtest=size(xtest,1);
        xtest= (xtest - ones(nbxtest,1)*meanX{t})./ (ones(nbxtest,1)*stdX{t} );
        Xtest{t}=xtest;
    else
        Xtest=[];
    end;
end
end