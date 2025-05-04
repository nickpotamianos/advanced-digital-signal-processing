function x = GenOMP(D, y, T0, epsilon, N)
%GENOMP Generalised Orthogonal Matching Pursuit (batch size N)
%   x = GENOMP(D, y, T0, epsilon, N) returns the sparse coefficients x that
%   approximate the signal y using the dictionary D.  Up to T0 atoms are
%   selected; the loop stops early if the residual energy falls below
%   EPSILON.  When N>1 the N best‑correlated atoms at each iteration are
%   added at once ("gOMP").
%
%   Compared to the original buggy version this implementation:
%     • NEVER calls an undefined helper like x_S on the workers – it is a
%       single, self‑contained function file that every worker can see.
%     • Has NO nested functions or anonymous handles that might vanish in
%       a parfor context.
%     • Works with or without the Parallel Computing Toolbox.
%
%   Inputs
%   ------
%   D        : (n × K) dictionary with **unit‑norm columns**
%   y        : (n × 1) signal (column vector)
%   T0       : maximum sparsity (positive integer)
%   epsilon  : residual energy tolerance (set 0 or [] to disable)
%   N        : atoms added per iteration (default 1 ➜ standard OMP)
%
%   Output
%   ------
%   x : (K × 1) sparse coefficient vector
%-----------------------------------------------------------------------
if nargin < 5 || isempty(N),       N = 1;       end
if nargin < 4 || isempty(epsilon), epsilon = 0; end
if nargin < 3 || isempty(T0),      T0 = size(D,2); end

% Pre‑computations -------------------------------------------------------
Dt = D';                   % (K × n) – used often
r  = y;                    % residual initialised to y
supp = false(1,size(D,2)); % logical support mask (faster than set ops)

% Main OMP / gOMP loop --------------------------------------------------
while true
    % Stop if residual target met or support is full
    if (epsilon > 0 && sum(r.^2) <= epsilon) || (nnz(supp) >= T0)
        break;    % finished
    end

    % 1) Correlate residual with all atoms
    correlations = abs(Dt * r);     % (K × 1)

    % 2) Pick the top‑N atoms not yet selected
    correlations(supp) = -inf;      % mask out existing support
    [~, idx] = sort(correlations, 'descend');
    newAtoms = idx(1:min(N, T0 - nnz(supp)));   % obey sparsity cap
    supp(newAtoms) = true;

    % 3) Solve least‑squares on the current support
    S  = find(supp);
    coefS = D(:,S) \ y;             % (|S| × 1)   – via QR / backslash

    % 4) Update residual
    r = y - D(:,S) * coefS;
end

% Build full‑length sparse vector ---------------------------------------
x = zeros(size(D,2),1);
if any(supp)
    x(find(supp)) = coefS; %#ok<FNDSB>
end
end
