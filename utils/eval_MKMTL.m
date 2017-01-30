function [eval,task_eval] = eval_MKMTL (Y, K, alpha, C, probY,evalType)
% Evaluation function for Multitask Multiple Kernel Learning models
% @param K: Testset Kernels with cell array of size task_num
% @param probY: cell array of size task_num, probability or confidence values for predictions if available,
% @param alpha: cell array of size task_num, weights computed for the
% training set
% empty if not - For classification tasks only
task_num = length(Y);
eval=0;
task_eval=zeros(task_num,1);

%%%%%%%%%%% Classification Eval Measures %%%%%%%%%%
if strcmp(evalType,'perfcurve')
    % Compute AUC of ROC curve
    ct=0;
    if isempty(probY)
        probY=cellfun(@(x,t) x*alpha{t}+C(t), K,num2cell(1:task_num),'UniformOutput',false);
    end
    for t = 1: task_num
        if(length(unique(Y{t}))==1)
            task_eval(t)= 0;
            continue;
        end
        [Xpr,Ypr,Tpr,AUCpr] = perfcurve(Y{t}, probY{t}, 1);%, 'xCrit', 'reca', 'yCrit', 'prec');
        
        if isnan(AUCpr)
            AUCpr=0;
        end
        task_eval(t)= AUCpr;
        eval= eval + AUCpr;
        ct=ct+1;
    end
    eval=eval/ct;
elseif strcmp(evalType,'prperfcurve')
    % Compute AUC of precision recall curve
    ct=0;
    if isempty(probY)
        probY=cellfun(@(x,t) x*alpha{t}+C(t), K,num2cell(1:task_num),'UniformOutput',false);
    end
    for t = 1: task_num
        if(length(unique(Y{t}))==1)
            task_eval(t)= 0;
            continue;
        end
        [Xpr,Ypr,Tpr,AUCpr] = perfcurve(Y{t}, probY{t}, 1, 'xCrit', 'reca', 'yCrit', 'prec');
        
        if isnan(AUCpr)
            AUCpr=0;
        end
        task_eval(t)= AUCpr;
        eval= eval + AUCpr;
        ct=ct+1;
    end
    eval=eval/ct;
elseif strcmp(evalType,'accuracy') % Multi-class accuracy measure
    if isempty(probY)
        probY=cellfun(@(x,t) x*alpha{t}+C(t), K,num2cell(1:task_num),'UniformOutput',false);
    end
    y=zeros(size(Y{1},1),1);
    for t = 1: task_num
        y=y+(((Y{t}+1)/2)*t);
    end
    [~,Ypred] = max(cell2mat(probY),[],2);
    corr=sum(Ypred==y);
    eval=corr/length(y);
    %{
    ct=0;
    for t = 1: task_num
        if(length(unique(Y{t}))==1)
            task_eval(t)= 0;
            continue;
        end
        if iscell(K)
            N = size(K{t},1);
        else
            N=size(K,1);
        end
        corr=sum(sign(probY{t})==Y{t});
        task_eval(t)= corr/N;
        eval=eval+corr/N;
        ct=ct+1;
    end
    eval=eval/ct;
    %}
elseif strcmp(evalType,'fmeasure')
    if isempty(probY)
        probY=cellfun(@(x,t) x*alpha{t}+C(t), K,num2cell(1:task_num),'UniformOutput',false);
    end
    ct=0;
    for t = 1: task_num
        if(length(unique(Y{t}))==1)
            task_eval(t)= 0;
            continue;
        end
        stats=confusionmatStats(Y{t},sign(probY{t}));
        task_eval(t)= sum(stats.Fscore)/2;
        eval=eval+sum(stats.Fscore)/2;
        ct=ct+1;
    end
    eval=eval/ct;
    %%%%%%%%%%% Regression Eval Measures %%%%%%%%%%
elseif strcmp(evalType,'rmse')
    % RMSE
    for t = 1: task_num
        Ypred = K{t} * alpha{t}+C(t);
        task_eval(t)=sqrt(mean((Ypred - Y{t}).^2));
    end
    task_eval(isinf(task_eval))=0;
    eval = mean(task_eval);
elseif strcmp(evalType,'mse')
    % MSE
    for t = 1: task_num
        Ypred = K{t} * alpha{t}+C(t);
        task_eval(t)=mean((Ypred - Y{t}).^2);
    end
    task_eval(isinf(task_eval))=0;
    eval = mean(task_eval);
    
elseif strcmp(evalType,'nmse')
    % NMSE
    for t = 1: task_num
        Ypred = (K{t} * alpha{t}+C(t));
        task_eval(t)=mean((Ypred - Y{t}).^2)/var(Y{t});
    end
    task_eval(isinf(task_eval))=0;
    eval = mean(task_eval);
elseif strcmp(evalType,'expvar')
    % Explained Variance
    for t = 1: task_num
        Ypred = (K{t} * alpha{t}+C(t));
        nmse=mean((Ypred - Y{t}).^2)/var(Y{t});
        task_eval(t)=1-nmse;
    end
    task_eval(isinf(task_eval))=0;
    eval = mean(task_eval);
else
    disp('Unknown error type. Please use one of the valid options.\n');
end



%%%%%% INbuilt Function %%%%%%%%%

    function stats = confusionmatStats(group,grouphat)
        % INPUT
        % group = true class labels
        % grouphat = predicted class labels
        %
        % OR INPUT
        % stats = confusionmatStats(group);
        % group = confusion matrix from matlab function (confusionmat)
        %
        % OUTPUT
        % stats is a structure array
        % stats.confusionMat
        %               Predicted Classes
        %                    p'    n'
        %              ___|_____|_____|
        %       Actual  p |     |     |
        %      Classes  n |     |     |
        %
        % stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
        % stats.precision = TP / (TP + FP)                  % for each class label
        % stats.sensitivity = TP / (TP + FN)                % for each class label
        % stats.specificity = TN / (FP + TN)                % for each class label
        % stats.recall = sensitivity                        % for each class label
        % stats.Fscore = 2*TP /(2*TP + FP + FN)            % for each class label
        %
        % TP: true positive, TN: true negative,
        % FP: false positive, FN: false negative
        %
        
        field1 = 'confusionMat';
        if nargin < 2
            value1 = group;
        else
            [value1,gorder] = confusionmat(group,grouphat);
        end
        
        numOfClasses = size(value1,1);
        totalSamples = sum(sum(value1));
        
        [TP,TN,FP,FN,accuracy,sensitivity,specificity,precision,f_score] = deal(zeros(numOfClasses,1));
        for class = 1:numOfClasses
            TP(class) = value1(class,class);
            tempMat = value1;
            tempMat(:,class) = []; % remove column
            tempMat(class,:) = []; % remove row
            TN(class) = sum(sum(tempMat));
            FP(class) = sum(value1(:,class))-TP(class);
            FN(class) = sum(value1(class,:))-TP(class);
        end
        
        for class = 1:numOfClasses
            accuracy(class) = (TP(class) + TN(class)) / totalSamples;
            sensitivity(class) = TP(class) / (TP(class) + FN(class));
            specificity(class) = TN(class) / (FP(class) + TN(class));
            precision(class) = TP(class) / (TP(class) + FP(class));
            f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
        end
        
        field2 = 'accuracy';  value2 = accuracy;
        field3 = 'sensitivity';  value3 = sensitivity;
        field4 = 'specificity';  value4 = specificity;
        field5 = 'precision';  value5 = precision;
        field6 = 'recall';  value6 = sensitivity;
        field7 = 'Fscore';  value7 = f_score;
        stats = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7);
        if exist('gorder','var')
            stats = struct(field1,value1,field2,value2,field3,value3,field4,value4,field5,value5,field6,value6,field7,value7,'groupOrder',gorder);
        end
        
    end
end
