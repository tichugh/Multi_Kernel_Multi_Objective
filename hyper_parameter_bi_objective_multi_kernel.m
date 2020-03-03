function f = hyper_parameter_bi_objective_multi_kernel(par,X,Y,kernels,CS_kernel)
    
    
    N = size(X,1); % number of stations
    K = Cov_matrix_calculations(X,par,kernels,CS_kernel) ;
    K = K + 1e-8 * eye(N,N); % N x N matrix
    TT = chol(K,'lower');
    K_inv = inv_chol(TT);    

    f_temp = Y'*K_inv*Y;   
    if det(K)==0
        f = [0.5*f_temp,0.5*log(1e-100)];
    else
        f = [0.5*f_temp,0.5*log(det(TT))];
    end
end