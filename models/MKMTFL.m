function [alpha,b,beta,varargout] = MKMTFL(M,Y,C,p,opts)
%% MKMTFL Multi-task Multiple Kernel Feature Learning
% written by Pratik Jawanpuria (Contact: pratik.jawanpuria@uni-saarland.de)
%
% Solve the following objective function
%
% $$\min_{\gamma \in \Delta_k} \max_{\lambda_j \in \Delta_{T,\tilde{p}}} \max_{\alpha_t \in S_{m_t}(C)} \sum_{t=1}^T \Big\{ \mathbf{1}^\top \alpha_t - \frac{1}{2}\alpha_t^\top \mathbf{Y}_t \big[ \sum_{j=1}^k \frac{\gamma_j \mathbf{K}_{tj}}{\lambda_{tj}}\big] \mathbf{Y}_t \alpha_t\Big\} $$
%
% where $\alpha_t$ is the Lagragian multiplier for the task $t$,
% $M$ and $Y$ are the cell array of size K,
% M is the cell array of length T, with each cell contains kernel matrix (NtxNt) of
% a single task
% $\mathcal{L}$ is the loss function (given by opts.loss),
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
%  @inproceedings{jawanpuria2011multi,
%  title={Multi-task multiple kernel learning},
%  author={Jawanpuria, Pratik and Nath, J Saketha},
%  booktitle={Proceedings of the 2011 SIAM International Conference on Data Mining},
%  year={2011},
%  organization={SIAM}
%  }





maxIter=opts.maxIter;
maxInIter=opts.maxIter;
tol=opts.tol;
wTol=opts.wTol; % Weight tolerence
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


if(p<2)
    error('Invalid value of p\n');
elseif(p==2)
    q = inf;
elseif(p==inf)
    q = 1;
else
    q = p/(p-2);
end
gamma = (1/K)*ones(K,1); % Initialize gamma
lambda = (1/(T^(1/q)))*ones(K,T); % Initialize lambda
if p==2
    lambda = ones(K,T); % unweighted
end


models=cell(1,T);
alpha=cell(1,T);
G = zeros(K,T);
stepSize=1e-6;

obj=0;%realmax('double');
b=zeros(1,T);
for it=1:maxIter % Out Iteration: optimization over gamma
    objValIn=0;%realmax('double');
    for in=1:maxInIter % In Iteration: optimization over lambda
        objValInOld=objValIn;
        % Solve for SVM/KRR given the Mcomb
        sumAlpha=0;
        for tt=1:T
            taskWeights=gamma./lambda(:,tt);
            taskWeights(isinf(taskWeights)) = 0;
            taskWeights(isnan(taskWeights)) = 0;
            if iscell(M)
                Mt=M{tt};
            else
                Mt=M;
            end
            Mcomb = weightSumKernels( Mt,taskWeights);
            if strcmp(loss,'reg')
                alpha{tt} = kernelRegression(Mcomb,Y{tt},1/C);
                sumAlpha=sumAlpha+alpha{tt}'*Y{tt}-(1/(2*C))*alpha{tt}'*alpha{tt};
                % Compute the Gram matrix
                for j=1:K
                    G(j,tt) = 0.5*alpha{tt}'*(Mt(:,:,j))*alpha{tt};%-0.5*alpha{tt}'*diag(diag(M{tt}(:,:,j)))*alpha{tt};
                    if G(j,tt)<0
                        G(j,tt)=0;
                    end
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
        gammaG=G.*repmat(gamma,1,T);
        gammaGlambda=gammaG./lambda;
        gammaGlambda(isinf(gammaGlambda)) = 0;
        gammaGlambda(isnan(gammaGlambda)) = 0;
        objValIn=sumAlpha-sum(sum(gammaGlambda));
        
        % Solve for lambda
        if(p~=2) % Compute lambda in dual space
            for j=1:K
                lambda(j,:) = G(j,:).^(1/(q+1))/((sum(G(j,:).^(q/(q+1))))^(1/q));
            end
        end
        lambda(lambda<wTol) = 0;
        if ~isreal(lambda)
            lambda=real(lambda);
            %fprintf('lambda not real');
        end
        relObj = (objValIn-objValInOld)/objValInOld;
        if mod(in,1)==100 && debugMode
            fprintf('InIteration %d, Objective:%f, Relative Obj:%f \n',in,objValIn,relObj);
        end
        
        %%%% Stopping Criteria
        if (abs(relObj) <= tol)
            break;
        end
    end
    
    gradGamma=-sum((G./lambda),2); % 1xK gradient vector
    gradGamma(isinf(gradGamma)) = 0;
    gradGamma(isnan(gradGamma)) = 0;
    objVal=sumAlpha + gamma'*gradGamma;
    
    % Compute stepSize
    %stepSize=sqrt(log(K))./(sqrt(it)*norm(gradGamma,inf));
    pvec = gradGamma.*stepSize-1-log(gamma);
    gamma = exp(-pvec);
    gamma = gamma/sum(gamma);
    gamma(gamma<wTol) = 0;
    
    
    
    obj=[obj;objVal];
    relObj = (obj(end)-obj(end-1))/obj(end-1);
    if relObj>0
        stepSize=stepSize*0.5;
    end
    if mod(it,1)==0 && debugMode
        fprintf('Iteration %d, Objective:%f, Relative Obj:%f \n',it,obj(end),relObj);
    end
    
    %%%% Stopping Criteria
    if (abs(relObj) <= tol)
        break;
    end
end
beta=repmat(gamma,1,T)./lambda;
beta(isinf(beta)) = 0;
beta(isnan(beta)) = 0;

% Get compact output notation
nout = max(nargout,1) - 3;
if nout==1
    varargout{1} = models; % model parameters from SVM
else
    varargout{1} = lambda; % task-specific weights
    varargout{2} = gamma; % shared weights weights
    varargout{3} = models; % model parameters from SVM
    varargout{4} = obj; % Objective values
    
end

end


