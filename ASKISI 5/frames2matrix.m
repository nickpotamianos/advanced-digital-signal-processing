function [X, rows, cols] = frames2matrix(matData, nFrames)
% Μετατρέπει τα nFrames πρώτα καρέ σε πίνακα X (pixels × frames)
%
% Υποστηριζόμενες μορφές:
%   • struct που μέσα του έχει ΜΙΑ μεταβλητή (ό,τι επιστρέφει η load)
%   • cell array καρέ
%   • 3‑D array  (rows × cols × T)         — γκρι
%   • 4‑D array  (rows × cols × 3 × T)     — RGB

    % === 1. Ξετυλίγουμε το struct της load() ============================
    if isstruct(matData)
        fns = fieldnames(matData);
        if numel(fns)==1
            matData = matData.(fns{1});     % <-- ασφαλής μονοδιάστατη πρόσβαση
        else
            error('Το .mat περιέχει πολλές μεταβλητές — πείτε ποια να χρησιμοποιήσω.');
        end
    end

    % === 2. Μετατρέπουμε σε cell array καρέ ============================
    if iscell(matData)
        framesCell = matData;

    elseif isnumeric(matData)
        switch ndims(matData)
            case 3      % rows × cols × T  (γκρι)
                T         = size(matData,3);
                framesCell = squeeze(num2cell(matData, [1 2])).';
                % χρησιμοποιούμε .’ ώστε το cell να βγαίνει 1×T αντί T×1
            case 4      % rows × cols × 3 × T  (RGB)
                matData   = rgb2gray(permute(matData,[1 2 4 3])); % -> rows×cols×T
                T         = size(matData,3);
                framesCell = squeeze(num2cell(matData, [1 2])).';
            otherwise
                error('Μη αναμενόμενες διαστάσεις βίντεο (%d).', ndims(matData));
        end

    else
        error('Μη υποστηριζόμενος τύπος δεδομένων (%s).', class(matData));
    end

    % === 3. Κατασκευή πίνακα X =========================================
    total  = min(nFrames, numel(framesCell));
    f0     = framesCell{1};
    if size(f0,3)==3, f0 = rgb2gray(f0); end
    rows   = size(f0,1);  cols = size(f0,2);

    X = zeros(rows*cols, total, 'double');
    for k = 1:total
        f = framesCell{k};
        if size(f,3)==3, f = rgb2gray(f); end
        X(:,k) = double(f(:));
    end
end
