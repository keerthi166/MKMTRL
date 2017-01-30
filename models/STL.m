function [alpha,b,models,obj] = STL(X,Y,C,opts)
%% Single Task Learning with SVM or Kernel Ridge Regression
%
% Solve the following objective function
%
% $$ \max_{0<\alpha_t<C} \sum_{t=1}^T \sum_{j=1}^k \Big\{ \mathbf{1}^\top \alpha_t - \frac{1}{2}\alpha_t^\top \mathbf{Y}_t  \mathbf{K}_{tj} \mathbf{Y}_t \alpha_t\Big\} $$
%
% where $\alpha_t$ are the lagragian multipliers for task $t$,
% $X$ and $Y$ are the cell array of size K,
% $\mathcal{L}$ is the loss function (given by opts.loss).
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


loss=opts.loss;
debugMode=opts.debugMode;
isKernelInput=false;
if isfield(opts,'isKernelInput')
    isKernelInput=opts.isKernelInput; % Whetther the input data is in kernel form
end
useClassWeights=true;
if isfield(opts,'useClassWeights')
    useClassWeights=opts.useClassWeights; % Use class weight option for libSVM
end

T=length(Y);

N=zeros(T,1);
svm_opt=cell(T,1);
for t=1:T
    N(t) = size(Y{t},1);    %No. examples per task
    Npos = sum(Y{t}==1);
    
    if strcmp(loss,'class') && (Npos==0)
        % randomly flip a label if no positive labels
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



models=cell(1,T);
alpha=cell(1,T);
G = zeros(1,T);

% Solve for SVM/KRR given the Mcomb
sumAlpha=0;
b=zeros(1,T);
for tt=1:T
    if isKernelInput
        if iscell(X)
            K=X{tt};
        else
            K=X;
        end
    else
        K=X{tt}*X{tt}';
    end
    
    if strcmp(loss,'reg')
        % Solve max_{\alpha} -lambda/2 alpha'*alpha +
        % alpha'*y-0.5*alpha'*K*alpha
        % alpha=inv(K+lambda*I)*y
        % We use lambda=1/C, set by CV
        
        alpha{tt} = kernelRegression(K,Y{tt},1/C);
        sumAlpha=sumAlpha+alpha{tt}'*Y{tt}-(1/(2*C))*alpha{tt}'*alpha{tt};
        % Compute the Gram matrix
        G(tt) = 0.5*alpha{tt}'*K*alpha{tt};%-0.5*alpha{tt}'*diag(diag(M{tt}(:,:,j)))*alpha{tt};
    elseif strcmp(loss,'class')
        models{tt} = svmtrain(Y{tt},[(1:N(tt))' K],svm_opt{tt});
        b(tt)=-models{tt}.rho;
        alpha{tt}=zeros(length(Y{tt}),1);
        alpha{tt}(full(models{tt}.SVs))=abs(models{tt}.sv_coef);
        alphay=alpha{tt}.*Y{tt}; % alpha.*y for classification
        sumAlpha=sumAlpha+sum(alpha{tt});
        % Compute the Gram matrix
        G(tt) = 0.5*alphay(models{tt}.SVs)'*K(models{tt}.SVs,models{tt}.SVs)*alphay(models{tt}.SVs);% -0.5*(alphay'*diag(M{tt}(models{tt}.SVs,models{tt}.SVs,j)).*alphay);
    else
        error('Invalid task function, it can be either class or reg.');
    end
end
obj=sumAlpha-sum(G);
end

