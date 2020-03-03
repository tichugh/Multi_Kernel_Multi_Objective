function [PS,PF] = EMO_multi_kernel(X,Y,kernels,Generations,N,CS_kernel)

    M = 2; % number of objectives
    no_var = size(X,2);
    
    n_weights = length(kernels);
    
    %% lower and upper bounds of parameters to be estimated
    lb_length_scale = 1e-3*ones(1,length(kernels)*no_var); ub_length_scale = 10*ones(1,length(kernels)*no_var);
    lb_signal_var = 1e-3*ones(1,length(kernels)); ub_signal_var = 10*ones(1,length(kernels));
    lb_noise = 1e-7; ub_noise = 1;
    lb_period = 1e-7*ones(1,no_var); ub_period = 100*ones(1,no_var);
    lb_weights = 1e-7*ones(1,n_weights); ub_weights = ones(1,n_weights);
    %%
    
    lb = [lb_length_scale,lb_signal_var,lb_noise,lb_period,lb_weights];
    ub = [ub_length_scale,ub_signal_var,ub_noise,ub_period,ub_weights];

    Boundary = [ub;lb];
    
    
    D = size(Boundary,2); % number of variables in selecting parameters for Kriging models
    MaxValue = Boundary(1,:);
    MinValue = Boundary(2,:);
    Population = rand(N,D);
    Population = Population.*repmat(MaxValue,N,1)+(1-Population).*repmat(MinValue,N,1); % generating random population
%     Population = round(Population);
    
%% normalized the weights so that sum of them is equal to 1

    weight_vectors = Population(:,end-length(kernels)+1:end);

    for w= 1:size(weight_vectors,1)
        weight_vectors(w,:) = weight_vectors(w,:)/sum(weight_vectors(w,:));
    end
    
    Population(:,end-length(kernels)+1:end) = weight_vectors;

    Population = unique(Population,'rows');
      
    %%
    Y = Y-mean(Y);
    FunctionValue = zeros(size(Population,1),M);
    for i = 1:size(Population,1)
        FunctionValue(i,:) = hyper_parameter_bi_objective_multi_kernel(Population(i,:),X,Y,kernels,CS_kernel);     
    end

    Archive = [Population,FunctionValue]; % storing all evaluated solutions
    
    FrontValue = P_sort(FunctionValue);
    CrowdDistance = F_distance(FunctionValue,FrontValue); % crowding distance calculation for NSGA-II

    ref_point = max(FunctionValue)+0.0001;
%     ref_point = [1,1];
    
    non_current = P_sort(FunctionValue,'first')==1;
    PF_current = FunctionValue(non_current,:);
    HV_old = P_evaluate_hv('HV',PF_current,ref_point);
    
    for Gene = 1 : Generations 
%         Gene 
        MatingPool = F_mating(Population,FrontValue,CrowdDistance,N); % mating pool
        Offspring = P_generator(MatingPool,Boundary,'Real',N);
        
         weight_vectors = Offspring(:,end-length(kernels)+1:end);

%         weight_vectors = Offspring(:,end-length(kernels)+2:end);

        for w= 1:size(weight_vectors,1)
            weight_vectors(w,:) = weight_vectors(w,:)/sum(weight_vectors(w,:));
        end
    
%         Offspring(:,end-length(kernels)+2:end) = weight_vectors;
        Offspring(:,end-length(kernels)+1:end) = weight_vectors;
        
        
        Offspring = unique(Offspring,'rows');

        check_r = ismember(Archive(:,1:end-M),Offspring,'rows');
        off1 = Archive(check_r,1:end-M);
        fit1 = Archive(check_r,end-M+1:end);

        check_r2 = ~ismember(Offspring,off1,'rows');

        off2 = Offspring(check_r2,:);
 

        fit2 = zeros(size(off2,1),M);
        for i  = 1:size(off2,1)
            fit2(i,:) = hyper_parameter_bi_objective_multi_kernel(off2(i,:),X,Y,kernels,CS_kernel);
        end
        
        Offspring = [off1;off2];
        Fitness = [fit1;fit2];

        Archive = [Archive;[Offspring,Fitness]];
        Population = [Population;Offspring];
        FunctionValue = [FunctionValue;Fitness];
%         FunctionValue = P_objective('value',Problem,M,Population);

        
        
        [FrontValue,MaxFront] = P_sort(FunctionValue,'half');
        CrowdDistance = F_distance(FunctionValue,FrontValue);

              
        Next = zeros(1,N);
        NoN = numel(FrontValue,FrontValue<MaxFront);
        Next(1:NoN) = find(FrontValue<MaxFront);
        
       
        Last = find(FrontValue==MaxFront);
        [~,Rank] = sort(CrowdDistance(Last),'descend');
        Next(NoN+1:N) = Last(Rank(1:N-NoN));
        
       
        Population = Population(Next,:);
        FrontValue = FrontValue(Next);
        CrowdDistance = CrowdDistance(Next);
        FunctionValue = FunctionValue(Next,:);
        
        if mod(Gene,10)==0
            non_current = P_sort(FunctionValue,'first')==1;
            PF_current = FunctionValue(non_current,:);
            HV_new = P_evaluate_hv('HV',PF_current,ref_point);
            
            if abs(HV_new - HV_old) <1e-3
                break;
            else
                HV_old = HV_new;
            end
        end
        
        
        
    end
    non = P_sort(Archive(:,end-M+1:end),'first')==1;
    PS = Archive(non,1:end-M);
    PF = Archive(non,end-M+1:end);
    
    Ideal = min(PF); Nadir = max(PF);
    PF = (PF - Ideal)./(Nadir - Ideal);
    
    


end