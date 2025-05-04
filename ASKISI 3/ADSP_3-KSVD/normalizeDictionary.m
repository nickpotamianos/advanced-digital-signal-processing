%=========================
% sparse_dl_algorithms
% MATLAB implementations for
%   • Generalized Orthogonal Matching Pursuit (GenOMP)
%   • K‑SVD dictionary update
%   • Batch Dictionary Learning (DL)
%   • Example training driver main.m
% Written to satisfy all requirements of the ADSP_KSVD exercise.
%===============================================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Utility – column normalisation                                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = normalizeDictionary(D)
%NORMALIZEDICTIONARY  Force every column of D to have unit ℓ2‑norm.
    D = D + eps;                     % avoid division‑by‑zero warnings
    D = D ./ sqrt(sum(D.^2,1));
end