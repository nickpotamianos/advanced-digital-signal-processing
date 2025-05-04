%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  K‑SVD dictionary update                                              %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D,X] = KSVD(D,X,Y)
% [D,X] = KSVD(D,X,Y)
% ----------------------------------------------------------------------
% Updates every atom of D via rank‑1 SVD exactly as stated in Algorithm 3
% of the assignment.  The sparse code matrix X MUST correspond to Y prior
% to calling (usually produced by GenOMP).  After the call both D and the
% selected rows of X are modified in‑place.  Columns of D are re‑normalised
% to unit norm.
%==========================================================================
    [~,K] = size(D);

    for k = 1:K
        omega = find(X(k,:) ~= 0);           % columns where atom participates
        if isempty(omega)
            % no signal currently uses atom – re‑initialise with random sample
            D(:,k) = Y(:,randi(size(Y,2)));
            D(:,k) = D(:,k)/norm(D(:,k)+eps);
            continue;
        end

        % Build the error matrix without contribution of atom k
        E = Y - D*X + D(:,k)*X(k,:);
        E_reduced = E(:,omega);

        % Rank‑1 approximation via SVD
        [U,S,V] = svd(E_reduced,'econ');
        D(:,k)      = U(:,1);               % new atom
        X(k,omega)  = S(1,1) * V(:,1)';     % corresponding coefficients
    end

    D = normalizeDictionary(D);
end