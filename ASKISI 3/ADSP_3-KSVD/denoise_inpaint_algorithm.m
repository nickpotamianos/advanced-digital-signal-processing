%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Denoising/Inpainting Algorithm                                      %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y_star] = denoise_inpaint_algorithm(D, Y, T0, epsilon, N)
% [Y_star] = denoise_inpaint_algorithm(D, Y, T0, epsilon, N)
% ----------------------------------------------------------------------
% Implements the denoising/inpainting algorithm as specified in the
% assignment.
%
% Input:
%  D        : (n×K) dictionary with unit-norm columns
%  Y        : (n×Nsig) input signals (possibly noisy or with missing values)
%  T0       : target sparsity level passed to GenOMP
%  epsilon  : residual threshold passed to GenOMP
%  N        : (optional) number of atoms per GenOMP iteration
% 
% Output:
%  Y_star   - reconstructed signals
% ----------------------------------------------------------------------
    if nargin < 5, N = 1; end
    [~, K] = size(D);
    [~, Nsig] = size(Y);
    X = zeros(K, Nsig);

    % -------- Sparse coding of each signal column --------
    for i = 1:Nsig
        X(:,i) = GenOMP(D, Y(:,i), T0, epsilon, N);
    end

    % -------- Reconstruction from sparse codes --------
    Y_star = D * X;
end