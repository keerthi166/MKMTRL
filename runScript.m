% Run Script for testing Multi-Task Multiple Kernel Relationship Learning
% on Stock2004 financial dataset
% @Author: Keerthiram Murugesan
% @Contact: kmuruges@cs.cmu.edu

% Some of the code is written by (O4/O6/2000) A. Rakotomamonjy for
% SimpleMKL (mentioned in the source code)
% MKMTFL Code is written by Pratik Jawanpuria (pratik.jawanpuria@uni-saarland.de)


clear;
addpath(genpath('../lib/libsvm-3.14/')) % Update libsvm if the problem is binary classification tasks

rng('default');

% Read Stock data
load('data/financial/stock04.mat')
% X - training set observation matrix
% Xt - test set observation matrix
% Y - training set response matrix
% Yt - test set response matrix

T= size(Y,2);
P=T;
N=size(Y,1);
Nrun=1;


% CV Settings
kFold = 5; % 5 fold cross validation


% Model Settings
models={'STL','MKL','MKMTFL','MKMTRL'}; 


% Add Intercept
Xtrain=cell(1,T);
Xtest=cell(1,T);
for tt=1:T
    Xtrain{tt}=[ones(size(X,1),1) X];
    Xtest{tt}=[ones(size(Xt,1),1) Xt];
end
Ytrain=mat2cell(Y,N,ones(1,T));
Ytest=mat2cell(Yt,N,ones(1,T));


%% Kernel Settings
kopts.efficientkernel=0;         % use efficient storage of kernels

kernelType={'gaussian','poly'};
kernelOpts={[1e-4,1],[1]};
variableType={'single', 'all'};

kernelInfo=createKernelBuildInfo(kernelType,kernelOpts,variableType,P);

nKernels=length(kernelInfo);

opts.loss='reg'; % Choose one: 'class', 'reg'
opts.scoreType='mse'; % Choose one: 'perfcurve', 'class', 'mse', 'nmse'
opts.debugMode=false;
opts.tol=1e-8; % tolerence parameter for optimization accuracy
opts.wTol=1e-8; % tolerance param for kernel weight thresholding
opts.maxIter=100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Run Experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result=cell(length(models),1);
for m=1:length(models)
    model=models{m};
    opts.model=model;
    
    
    % Initilaization
    result{m}.score=zeros(Nrun,1);
    result{m}.taskScore=zeros(T,Nrun);
    
    
    runtime=0;
    % Run Id - For Repeated Experiment
    for rId=1:Nrun
        
        % Normalize Kernel
        normM=cell(1,T);
        Mtrain=cell(1,T);
        [Xtrain,~,meanX,stdX] = normalizeMultitaskData(Xtrain);
        if strcmp(model,'STL')
            for tt=1:T
                Mtrain = constructMTMKLKernel(Xtrain,createKernelBuildInfo({'polyhomog'},{1},{'all'},P),kopts);
            end
        else
            Mtrain = constructMTMKLKernel(Xtrain,kernelInfo,kopts);
            for tt=1:T
                [Mtrain{tt},normM{tt}]=normalizeKernel(Mtrain{tt});
            end
        end
        
       
        %------------------------------------------------------------------------
       %%                   Construct Models
        %------------------------------------------------------------------------
        opts.C=1e+1;
        opts.p=3;
        opts.mu=1e-5;
        
        tic
        beta=zeros(nKernels,T);
        switch model
            case 'STL'
                % Single Task Learning
                [alpha,b,task_models,obj] = STL(Mtrain,Ytrain,opts.C,opts);
            case 'MKL' 
                % Multiple Kernel Learning
                [alpha,b,beta,task_models,obj] = MKL(Mtrain,Ytrain,opts.C,opts.p,opts);
            case 'MKMTFL'
                % Multi-task Multiple Kernel Feature Learning
                [alpha,b,beta,lambda,gamma,task_models,obj] = MKMTFL(Mtrain,Ytrain,opts.C,opts.p,opts);
            case 'MKMTRL'
                opts.wTol=1e-0;
                % Multi-task Multiple Kernel Feature Learning
                [alpha,b,beta,Omega,task_models,obj] = MKMTRL(Mtrain,Ytrain,opts.C,opts.mu,opts);
        end
        runtime=runtime+toc;
        %------------------------------------------------------------------------
       %%                   Eval Models
        %       Compute Area under the ROC curve & Accuracy
        %------------------------------------------------------------------------
        
        % Normalize Test Data
        [Xtest,~,~,~] = normalizeMultitaskData(Xtest,[],meanX,stdX);
        Mtest=cell(1,T);
        switch model
            case 'STL'
                Mtest = constructMTMKLKernel(Xtest,createKernelBuildInfo({'polyhomog'},{1},{'all'},P),kopts,Xtrain,num2cell(ones(1,T)));
            case 'MTFL'
                for tt=1:T
                    Mtest{tt}=Xtest{tt}*(inv(D)\Xtrain{tt}');
                end
            case 'MTRL'
                Mtest = constructMTMKLKernel(Xtest,createKernelBuildInfo({'polyhomog'},{1},{'all'},P),kopts,Xtrain,num2cell(ones(1,T)));
            case {'MKL','MKMTFL','MKMTRL'}
                weights=cellfun(@(t,nm) beta(:,t)./nm, num2cell(1:T),normM, 'uniformOutput', false);
                Mtest = constructMTMKLKernel(Xtest,kernelInfo,kopts,Xtrain,weights);
                
        end
        
        [result{m}.score(rId),result{m}.taskScore(:,rId)]=eval_MKMTL(Ytest, Mtest, alpha, b,[], opts.scoreType);
        runtime=runtime+toc;
        if(opts.debugMode)
            fprintf('Method: %s, RunId: %d, %s: %f \n',model,rId,opts.scoreType,result{m}.score(rId));
        end
    end
    %------------------------------------------------------------------------
    %%                   Collect & Store Model Stats
    %------------------------------------------------------------------------
    result{m}.model=model;
    result{m}.loss=opts.loss;
    result{m}.scoreType=opts.scoreType;
    result{m}.opts=opts;
    result{m}.meanScore=mean(result{m}.score);
    result{m}.stdScore=std(result{m}.score);
    result{m}.meanTaskScore=mean(result{m}.taskScore,2);
    result{m}.stdTaskScore=std(result{m}.taskScore,0,2);
    result{m}.runtime=runtime/Nrun;
    fprintf('Method: %s, %s: %f. Runtime: %0.4f\n', opts.model,opts.scoreType,result{m}.meanScore,result{m}.runtime);
end



