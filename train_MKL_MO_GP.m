function solutions = train_MKL_MO_GP(X,Y,kernels,CS_kernel)

addpath(genpath('Support_files'));
addpath(genpath('GP_files'));

maxgen = 50;
pop_size = 10;

[PS,PF] = EMO_multi_kernel(X,Y,kernels,maxgen,pop_size,CS_kernel);


solutions = [PS,PF];

end



