function [alpha,b,beta,varargout] = MKL(M,Y,C,p,opts)
%% MKL Multiple Kernel Learning Extension for Multi-task Framework using Lp-MKL
%
% p is the norm of the weight vectors ||W||_p
%
% See paper for more details on the notation.
%
% References: 
%
%  @inproceedings{murugesan2017multi,
%  title={Multi-task multiple kernel relationship learning},
%  author={Murugesan, Keerthiram and Carbonell, Jaime},
%  booktitle={Proceedings of the 2017 SIAM International Conference on Data Mining},
%  year={2017},
%  organization={SIAM}
%  }

maxIter=opts.maxIter;
tol=opts.tol;
loss=opts.loss;
debugMode=opts.debugMode;
useClassWeights=true;
if isfield(opts,'useClassWeights')
    useClassWeights=opts.useClassWeights; % Use class weight option for libSVM
end

T=length(Y);
if iscell(M)
    K=size(M{1},3);
else
    K=size(M,3);
end

N=zeros(T,1);
svm_opt=cell(T,1);
for t=1:T
    N(t) = size(Y{t},1);    %No. examples per task
    Npos = sum(Y{t}==1);
    if strcmp(loss,'class') && (Npos==0)
        % randomly flip a label
        s=randsample(N(t),1);
        Y{t}(s)=1;
        Npos=1;
    end
    if useClassWeights
        svm_opt{t}=sprintf('-q -t 4 -c %f -w1 %f -w-1 %f', C, (N(t)-Npos)/N(t), Npos/N(t));
    else
        svm_opt{t}=sprintf('-q -t 4 -c %f', C);
    end
end

% Initialization
beta = (1/sqrt(K))*ones(K,T); % Initialize beta


models=cell(1,T);
alpha=cell(1,T);
G = zeros(K,T);

obj=0;%realmax('double');
b=zeros(1,T);
for it=1:maxIter
    % Solve for SVM given the Mcomb
    sumAlpha=0;
    for tt=1:T
        if iscell(M)
            Mt=M{tt};
        else
            Mt=M;
        end
        Mcomb = weightSumKernels( Mt,beta(:,tt));
        if strcmp(loss,'reg')
            alpha{tt} = kernelRegression(Mcomb,Y{tt},1/C);
            sumAlpha=sumAlpha+alpha{tt}'*Y{tt}-(1/(2*C))*alpha{tt}'*alpha{tt};
            % Compute the Gram matrix
            for j=1:K
                G(j,tt) = 0.5*alpha{tt}'*(Mt(:,:,j))*alpha{tt};%-0.5*alpha{tt}'*diag(diag(M{tt}(:,:,j)))*alpha{tt};
            end
            b=zeros(1,T);
        elseif strcmp(loss,'class')
            models{tt} = svmtrain(Y{tt},[(1:N(tt))' Mcomb],svm_opt{tt});
            b(tt)=-models{tt}.rho;
            alpha{tt}=zeros(length(Y{tt}),1);
            alpha{tt}(full(models{tt}.SVs))=abs(models{tt}.sv_coef);
            alphay=alpha{tt}.*Y{tt}; % alpha.*y for classification
            sumAlpha=sumAlpha+sum(alpha{tt});
            % Compute the Gram matrix
            for j=1:K
                G(j,tt) = 0.5*alphay(models{tt}.SVs)'*Mt(models{tt}.SVs,models{tt}.SVs,j)*alphay(models{tt}.SVs);% -0.5*(alphay'*diag(M{tt}(models{tt}.SVs,models{tt}.SVs,j)).*alphay);
            end
        else
            error('Invalid task function, it can be either class or reg.');
        end
    end
    betaG=G.*beta;
    objVal=sumAlpha-sum(sum(betaG));
    
    % Solve for beta
    sqW=betaG.*beta;
    sqW=max(sqW,0);
    %normW=sum(sqW.^(p/(p+1))).^(1/p); % 1xT vector
    %beta=sqW.^(1/(p+1))./repmat(normW,K,1);
    for tt=1:T
        beta(:,tt) = sqW(:,tt).^(1/(p+1))/((sum(sqW(:,tt).^(p/(p+1))))^(1/p));
    end
    
    obj=[obj;objVal];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if mod(it,1)==0 && debugMode
        fprintf('Iteration %d, Objective:%f, Relative Obj:%f \n',it,obj(end),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= tol)
        break;
    end
end

% Get compact output notation
nout = max(nargout,1) - 3;
if nout==1
    varargout{1} = models; % model parameters from SVM
else
    varargout{1} = models; % model parameters from SVM
    varargout{2} = obj; % Objective values
    
end

end

