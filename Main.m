%% This is the function to train a GP using composite kernel.
% The copyrights are with Tinkle Chugh, t.chugh@exeter.ac.uk
% Please, see the licence file for more information and cite the following
% article if you use the code:
% Chugh T., Rahat A., Palar P.S. (2019) Trading-off Data Fit and Complexity 
% in Training Gaussian Processes with Multiple Kernels. In: Nicosia G., 
% Pardalos P., Umeton R., Giuffrida G., Sciacca V. (eds) Machine Learning, 
% Optimization, and Data Science. LOD 2019. Lecture Notes in Computer 
% Science, vol 11943. Springer, Cham

clear; clc; close all;

%% Two ways of composite kernel - weighted sum and weighted product % use one at one time
% CS_kernel = 'WS'; % weighted sum
CS_kernel = 'WP'; % weighted product

addpath(genpath('Support_files'));
addpath(genpath('NSGA-II_files'));
addpath(genpath('GP_files'));

kernels = {'RBF','EXP','M32','M5/2','periodic'}; % for the multiobjective and multi-kernel


%% load the data set
N = 200;
X = linspace(-10,10,N)';
Y = 1 + X*5e-2 + sin(X)./X + 0.2*randn(N,1);

%% Divide the data-set into k folds 
k = 2;
indices = crossvalind('Kfold',N,k);
r = find(indices==1);
X_select = X(r,:);
Y_select = Y(r,:);
X(r,:) = [];
Y(r,:) = [];
solutions = train_MKL_MO_GP(X,Y,kernels,CS_kernel);
figure;
scatter(-solutions(:,end-1),solutions(:,end),50);
[theta,f_vector] = select_solution(solutions,X,Y,X_select,Y_select,kernels,CS_kernel);
hold on;
scatter(-f_vector(1),f_vector(2),100,'filled');
hold off;
box on; grid on; 
xlabel('Normalised data fit'); ylabel('Normalised complexity');
ax = gca;
ax.FontSize = 14; 
ax.FontWeight = 'bold';
legend('Pareto front', 'Selected solution');


 
%% Prediction on the testing data

N = 100;
X_test = linspace(-10,10,N)';
Y_test = 1 + X*5e-2 + sin(X)./X + 0.2*randn(N,1);
% 
[Y_pred,std_pred] = prediction_multi_kernel(theta,X,Y,X_test,kernels,CS_kernel); % posterior mean with its standard deviation
figure;
scatter(X_test,Y_test);
hold on;
scatter(X_test,Y_pred);
hold off;
box on; grid on; 
xlabel('X'); ylabel('Y');
ax = gca;
ax.FontSize = 14; 
ax.FontWeight = 'bold';
legend('Testing', 'Prediction');


%%





