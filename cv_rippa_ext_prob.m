function [ep_best,val,timez] = cv_rippa_ext_prob(dsites,rhs,rbf,ep,n_folds,...
    the_norm,reduction_par)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The extension of the Rippa's method to k-fold CV for the tuning of the
% shape parameter, with a stochastic low-rank approximation for the inverse
% of the kernel matrix.
% This function is inspired by LOOCV2D.m by G. Fasshauer.
% Calls on: DistanceMatrix.m by G. Fasshauer
%
% Input
%   dsites: Mxs matrix representing a set of M data sites in R^s
%              (i.e., each row contains one s-dimensional point)
%   rhs: Mx1 matrix of function values at the dsites
%   rbf: function handle, the chosen RBF for the interpolation
%   ep: Dx1 matrix of values for the tuning of the shape parameter
%   n_folds: number of folds k for the k-fold CV
%   the_norm: chosen norm for the validation error
%   reduction_par: number in (0,1), it controls the dimension of the random
%                  matrix W, n x floor(n*reduction_par). Alternatively,
%                  integer number such that W is n x reduction_par.
%
% Output
%   ep_best: the optimal value of the shape parameter
%   val: the validation error
%   timez: the time employed by the k-fold CV process
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if reduction_par <= 1
    s = floor(size(dsites,1)*reduction_par);
else
    s = reduction_par;
end

rng(42)
folds = cvpartition(length(rhs),'KFold',n_folds);

DM_data = DistanceMatrix(dsites,dsites);
maxEF = zeros(length(ep),1);

tic

for i=1:length(ep)
    
    EF = zeros(length(rhs),1);
    IM = rbf(ep(i),DM_data);
        
    rng(42)
    W = randn(size(dsites,1),s);  

    V = IM*W;
    invIM = W*((V'*V)\eye(s))*V';

    coeffs = invIM*rhs;
            
    if n_folds == size(dsites,1)
              
        EF = coeffs./diag(invIM);       
       
    else
                
        pos = 1;
        
        for j=1:n_folds
            
            test_ind = test(folds,j);
            invIM_loc = invIM(test_ind,test_ind);   

            EF(pos:pos+sum(test_ind)-1) = invIM_loc\coeffs(test_ind);

            pos = pos + sum(test_ind);
        end        
                
    end
    
    maxEF(i) = norm(EF(:),the_norm);


end

ep_best = ep(maxEF==min(maxEF));

if length(ep_best)>1
    idx=randperm(length(ep_best),1);
    ep_best = ep_best(idx);    
end
val = min(maxEF);
timez = toc;