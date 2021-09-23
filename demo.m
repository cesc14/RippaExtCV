%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Script that perform the parameter tuning in RBF interpolation using the
% extended Rippa's scheme. To use this script, please cite:
% F. Marchetti, "The extension of the Rippa's algorithm beyond LOOCV",
% submitted.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all

% The interpolation dataset

[xx,yy] = meshgrid(linspace(-1,1,30));
dsites = [xx(:) yy(:)];

% The vector for the shape parameter tuning
mine = 0.01; maxe = 0.5; ne = 101;
ep = linspace(mine,maxe,ne);

% The chosen norm

the_norm = inf;

% The number k of k-fold CV

n_folds = 10;

% The test function and function values at the data sites

f = @(x,y) sin(x)./(x.^2+1) .* cos(y)./(y.^2+1);
rhs = f(dsites(:,1),dsites(:,2));  

% The chosen RBF

rbf = @(e,r) max(0,1-e.*r).^4.*(4.*(e.*r) + 1); %Wendland c2

% The extended Rippa's algorithm

[ep_best,val,timez]=cv_rippa_ext(dsites,rhs,rbf,ep,n_folds,the_norm);
    

fprintf('Best parameter value: %f\n' , ep_best)
fprintf('Validation error: %f\n' , val)
fprintf('Time employed: %f\n' , timez)

    
