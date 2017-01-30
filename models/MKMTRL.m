function [alpha,b,beta,varargout] = MKMTRL(M,Y,C,lambda,opts)
%% MKMTRL Multiple Kernel Multi-task Relationship Learning
%
% Solve the following objective function
%
% $$\min_{\mathbf{\Omega},\mathbf{B} \geq 0} \max_{0 \leq \alpha \leq C}
% h(\alpha,\mathbf{B}) +
% \frac{\mu}{2}tr(\mathbf{B}\mathbf{\Omega}^{-1}\mathbf{B}^\top) $$
% s.t.,  
% $$ \mathbf{\alpha}_t^\top \mathbf{y}_t = 0, $$
% $$ \mathbf{\Omega} \succeq 0,$$
% $$ tr(\mathbf{\Omega}) \leq 1 $$ 
%
% where,
% $$ h(\alpha,\mathbf{B})=\sum_{t=1}^T  \Big\{\mathbf{1}^\top
% \mathbf{\alpha}_t -\frac{1}{2} \mathbf{\alpha}_t^\top \mathbf{Y}_t
% \Big(\sum_{k=1}^K \beta_{tk}\mathbf{\mathcal{K}}_{tk}\Big) \mathbf{Y}_t
% \mathbf{\alpha}_t \Big\} $$
%
% where $\alpha_t$ is the Lagragian multiplier for the task $t$,
% $M$ and $Y$ are the cell array of size K,
% M is the cell array of length T, with each cell contains kernel matrix (NtxNt) of
% a single task
% $\mathcal{L}$ is the loss function (given by opts.loss),
% $\Omega$ is the task relationship (KxK) matrix.
% $\mu$ is the regularization parameter,
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
maxInIter=opts.maxIter;
maxBetaIter=opts.maxIter;
tol=opts.tol;
wTol=opts.wTol;
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

Omega = eye(T)/T;
epsilon=1e-8;


models=cell(1,T);
alpha=cell(1,T);
G = zeros(K,T);

obj=0;%realmax('double');
b=zeros(1,T);
for it=1:maxIter % Out Iteration: optimization over Omega
    objValIn=0;%realmax('double');
    for in=1:maxInIter % In Iteration: optimization over beta
        objValInOld=objValIn;
        % Solve for SVM/KRR alpha given the Mcomb
        sumAlpha=0;
        b=zeros(1,T);
        for tt=1:T
            if iscell(M)
                Mt=M{tt};
            else
                Mt=M;
            end
            Mcomb = weightSumKernels( Mt,beta(:,tt));
            if strcmp(loss,'reg')
                alpha{tt} = kernelRegression(Mcomb,Y{tt},(1/C));
                sumAlpha=sumAlpha+alpha{tt}'*Y{tt}-(1/(2*C))*alpha{tt}'*alpha{tt};
                % Compute the Gram matrix
                for j=1:K
                    G(j,tt) = 0.5*alpha{tt}'*(Mt(:,:,j))*alpha{tt};%-0.5*alpha{tt}'*diag(diag(M{tt}(:,:,j)))*alpha{tt};
                end
                
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
        iOmega=inv(Omega);
        
        objValIn=sumAlpha-sum(sum(betaG)) + lambda*trace(beta*(iOmega)*beta');
        
        % Solve for beta
        converged=0;
        betaIter=1;
        sqW=betaG.*beta; %KxT matrix
        sqW=max(sqW,0);
        objValBeta=0;
        %sqWOmega=sqW*Omega; %KxT matrix
        %beta=(sqWOmega)/sqrt(trace(sqWOmega*sqW')); % closed-form solution
        
        while ~converged && betaIter<maxBetaIter
            gradNorm=0;
            objValBetaOld=objValBeta;
            betaOld=beta;
            for j=1:K
                sqWBeta=sqW(j,:)./(beta(j,:).^3);
                sqWBeta(isnan(sqWBeta))=0;
                Hj=lambda*iOmega'+2*diag(sqWBeta); % Hessian TxT matrix
                gradj=lambda*beta(j,:)*iOmega-sqW(j,:)./(beta(j,:).^2); % Gradient 1xT matrix
                gradj(isnan(gradj))=0;
                gradNorm=gradNorm+norm(gradj);
                warning off;
                beta(j,:) = beta(j,:) - gradj/Hj;
                warning on;
            end
            betaG=G.*beta;
            objValBeta=sumAlpha-sum(sum(betaG)) + lambda*trace(beta*(iOmega)*beta');
            relObj = (objValBeta-objValBetaOld)/objValBetaOld;
            if mod(it,1)==100 && debugMode
                fprintf('BetaIteration %d, Objective:%f, Relative Obj:%f gradNorm:%f betaNorm: %f\n',betaIter,objValBeta,relObj,gradNorm, norm(betaOld-beta));
            end
            if (abs(relObj) <= tol)
                break;
            end
            converged = gradNorm < tol;
            betaIter=betaIter+1;
        end
        
  
        beta(beta < wTol) = 0;
        beta(isnan(beta)) = 0;
        
        relObj = (objValIn-objValInOld)/objValInOld;
        if mod(in,1)==100 && debugMode
            fprintf('InIteration %d, Objective:%f, Relative Obj:%f \n',in,objValIn,relObj);
        end
        
        %%%% Stopping Criteria
        if (abs(relObj) <= tol)
            break;
        end
    end
    
    betaG=G.*beta;
    objVal=sumAlpha-sum(sum(betaG)) + lambda*trace(beta*(iOmega)*beta');
    
    % Solve for Omega
    temp=beta'*beta;
    [eigVector,eigValue]=eig(temp+epsilon*eye(T));
    clear temp;
    eigValue=sqrt(abs(diag(eigValue)));
    eigValue=eigValue/sum(eigValue);
    Omega=eigVector*diag(eigValue)*eigVector';
    
    
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
    varargout{1} = Omega; %task -relationship matrix
    varargout{2} = models; % model parameters from SVM
    varargout{3} = obj; % Objective values
    
end

end

