function k = multi_kernel_calculation(xi,xj,par,kernels,CS_kernel)

if xi==xj
    delta = 1;
else
    delta = 0;
end
no_var = size(xi,2);
n1 = length(kernels);


n_params = length(par); % it shound be length(kernels)*(no_var+1) + 1 +  length(kernels) + no_var 

if n_params~= n1*(no_var+1) + 1 + n1 + no_var
    error('check the number of parameters to be estimated');
end

sigmax_g = par(1:no_var*n1);
p1 = length(sigmax_g);
sigmaf_g = par(p1+1: p1 + n1);
p2 = length(sigmaf_g) + length(sigmax_g);
sigman = par(p2+1);
p3 = length(sigmaf_g) + length(sigmax_g) + length(sigman);
period_k = par(p3+1:p3 + no_var);
p4 = length(sigmaf_g) + length(sigmax_g) + length(sigman) + length(period_k);
w_vector = par(p4+1:end);
% sum(w_vector)
if (sum(w_vector)<0.99 && sum(w_vector)>1.01)
    error('sum of weight vectors must be equal to 1');
end

k_vector = zeros(1,length(kernels));

%% RBF
    sigmax = sigmax_g(1:no_var);
    sigmaf = sigmaf_g(1);
    s = 0;
    for s_var = 1:no_var
        r = pdist2(xi(s_var),xj(s_var));
        s = s + r^2./sigmax(s_var).^2; 
    end
%     s = s^2;
    s = exp(-0.5*s); 
    k_vector(1) = sigmaf^2*s;

%% EXP
    sigmax = sigmax_g(no_var+1:2*no_var);
    sigmaf = sigmaf_g(2);
    s = 0;
    for s_var = 1:no_var
        r = pdist2(xi(s_var),xj(s_var));
        s = s + r./sigmax(s_var); 
    end
%     s = sqrt(s);
    s = exp(-s);
    k_vector(2) = sigmaf^2*s;
%% M32
    sigmax = sigmax_g(2*no_var+1:3*no_var);
    sigmaf = sigmaf_g(3);
    s = 0;
    for s_var = 1:no_var
        r = pdist2(xi(s_var),xj(s_var));
        s = s + r./sigmax(s_var); 
    end
%     s = sqrt(s);    
    k_vector(3) = sigmaf^2*(1+ sqrt(3)*s)*exp(-sqrt(3)*s);


%% M5/2
    sigmax = sigmax_g(3*no_var+1:4*no_var);
    sigmaf = sigmaf_g(4);
    s = 0;
    for s_var = 1:no_var
        r = pdist2(xi(s_var),xj(s_var));
        s = s + r./sigmax(s_var); 
    end
%     s = sqrt(s);    
    k_vector(4) = sigmaf^2*(1+ sqrt(5)*s +(5/3)*s^2)*exp(-sqrt(5)*s);

    
% k_vector(4) = sigmaf4^2*(s1)*(s2) ;
    
%% Periodic
    sigmax = sigmax_g(4*no_var+1:5*no_var);
    sigmaf = sigmaf_g(5);
    s = 0;
    for s_var = 1:no_var
        r = pdist2(xi(s_var),xj(s_var));
        s = s + sin(pi*r/period_k(s_var))^2/sigmax(s_var).^2;

    end
     k_vector(5) = sigmaf^2*exp(-2*s);  
     
     if strcmp(CS_kernel,'WS')
         k = sum(w_vector.*k_vector);
     elseif strcmp(CS_kernel,'WP')
         k = prod(w_vector.*k_vector);
     end

  
k = k + sigman.^2*delta;

end


