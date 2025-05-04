%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Dictionary Learning – batch scheme                                   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D,X,mseHist] = DictionaryLearning(D0,Y,T0,epsilon,noEpochs,N)
% [D,X,mseHist] = DictionaryLearning(D0,Y,T0,epsilon,noEpochs)       – OMP inner
% [D,X,mseHist] = DictionaryLearning(D0,Y,T0,epsilon,noEpochs,N)     – gOMP inner
% ----------------------------------------------------------------------
% INPUT
%  D0        : (n×K) initial dictionary (random or supplied)
%  Y         : (n×Nsig) training data (each column ℓ2‑normalized!)
%  T0        : target sparsity level passed to GenOMP
%  epsilon   : residual threshold passed to GenOMP
%  noEpochs  : total sweeps over all training signals
%  N         : (optional) number of atoms per GenOMP iteration
% 
% OUTPUT
%  D        – learned dictionary with unit‑norm columns
%  X        – final sparse code of training set
%  mseHist  – (noEpochs×1) reconstruction MSE after each epoch
% ----------------------------------------------------------------------
    if nargin<6, N = 1; end
    D = normalizeDictionary(D0);
    [~,K] = size(D);
    Nsig  = size(Y,2);
    X     = zeros(K,Nsig);
    mseHist = zeros(noEpochs,1);

    for epoch = 1:noEpochs
        % -------- Sparse coding step ------------------------
        parfor i = 1:Nsig
            X(:,i) = GenOMP(D,Y(:,i),T0,epsilon,N);
        end

        % -------- Dictionary update step --------------------
        [D,X] = KSVD(D,X,Y);

        % -------- Progress metric ---------------------------
        mseHist(epoch) = mean(sum((Y - D*X).^2,1));
        fprintf('Epoch %d/%d – MSE: %.4e\n',epoch,noEpochs,mseHist(epoch));
    end
end