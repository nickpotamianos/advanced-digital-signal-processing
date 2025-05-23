function seq = matrix2frames(X, rows, cols)
% Μετατρέπει τον πίνακα X (pixels × frames) σε cell array καρέ, 
% κλιμακώνοντας σωστά για εγγραφή με VideoWriter.
%
%  • Αν τα δεδομένα είναι στη ζώνη [0,1]  -> τα γράφουμε ως double.
%  • Αν είναι στη ζώνη [0,255]           -> τα γράφουμε ως uint8.
%  • Παίρνουμε clamp στα [0,255] μετά από SVD/PCA.

    nFrames = size(X,2);
    seq     = cell(1,nFrames);

    % Εκτίμηση κλίμακας από τον ΟΛΙΚΟ πίνακα (γρήγορο)
    globalMax = max(X(:));
    globalMin = min(X(:));

    useUint8 = globalMax > 1.5;     % τιμές >1 ⇒ μάλλον κλίμακα 0‑255

    for k = 1:nFrames
        f = reshape(X(:,k), rows, cols);

        % Clamp για να αποφύγουμε wrap‑around
        if useUint8
            f = max(min(f,255),0);           % 0–255
            seq{k} = uint8(f);
        else
            f = max(min(f,1),0);             % 0–1
            seq{k} = f;                      % double
        end
    end
end
