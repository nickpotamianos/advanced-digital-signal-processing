
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Dictionary visualisation – dispD                                     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dispD(D,blk)
% dispD(D,blk)  Display dictionary atoms as images on a single figure.
%   D   : (n×K) dictionary with unit‑norm columns, n = blk^2
%   blk : block (patch) edge length in pixels (default: sqrt(size(D,1)))

    if nargin<2,  blk = round(sqrt(size(D,1))); end
    K = size(D,2);
    cols = ceil(sqrt(K));
    rows = ceil(K/cols);

    % normalise each atom to [0 1] for display
    Dimg = D - min(D,[],1);
    Dimg = Dimg ./ max(Dimg,[],1);

    margin = 1; % pixels between atoms
    canvas = ones(rows*blk + (rows+1)*margin, cols*blk + (cols+1)*margin);

    k = 1;
    for r = 1:rows
        for c = 1:cols
            if k>K, break; end
            atom = reshape(Dimg(:,k),blk,blk);
            rr = (r-1)*(blk+margin) + margin + (1:blk);
            cc = (c-1)*(blk+margin) + margin + (1:blk);
            canvas(rr,cc) = atom;
            k = k+1;
        end
    end

    figure; imshow(canvas,[]); title(sprintf('Dictionary atoms (%d×%d)',blk,blk));
end