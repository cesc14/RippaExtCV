function [err,timez] = surrogate_cv(G,L,rhs,exc,n_folds)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% A surrogate k-fold CV algorithm for RBF collocation
%
% Calls on: DistanceMatrix.m by G. Fasshauer
%
% Input
%   G: MxN collocation matrix
%   L: MxN evaluation matrix
%   rhs: Mx1 matrix, the right-hand side of the collocation problem
%   exc: Mx1 matrix, the exact solution at the collocation points
%   n_folds: integer between 2 and M, folds the k-fold CV
%
% Output
%   err: Mx1 matrix, approximated validation error vector
%   timez: time (in seconds) employed by the surrogate k-fold CV process
%
% If you use this function, please cite:
% F. Marchetti, "A fast surrogate cross validation algorithm for meshfree
% RBF collocation approaches"
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rng(42);
folds = cvpartition(length(rhs),'KFold',n_folds);

tic

err = zeros(length(rhs),1);

invG = pinv(G); invL = pinv(L,1e-16);

c = invG*rhs;
    
for j=1:n_folds
    
    test_ind = test(folds,j);
    invG_loc = invG(test_ind,test_ind);  
    invinvL = pinv(invL(:,test_ind));
    tau = invG(:,test_ind)*(invG_loc\c(test_ind));
    err(test_ind) = invinvL*tau;

end        

f = L*c;

err = err - f + exc;

timez = toc;
