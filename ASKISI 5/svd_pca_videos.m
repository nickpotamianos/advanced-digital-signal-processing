%% 1. Φόρτωση βίντεο και επιλογή 800 καρέ
videos   = {'boostrap','campus','lobby'};
nFrames  = 800;
rankList = [10 3];       % βαθμίδες κρατούμενων ιδ. τιμών
batchSz  = 100;

for v = 1:numel(videos)
    rawData = load([videos{v} '.mat']);    % μπορεί να έχει 1 ή πολλές μεταβλητές
    [X, r, c] = frames2matrix(rawData, nFrames);

    %% 2–3. Επεξεργασία ολόκληρης της ακολουθίας (rank‑10 & rank‑3)
    for rnk = rankList
        Xhat = lowrank_reconstruct(X, rnk);
        reconSeq = matrix2frames(Xhat, r, c);
        % (προβολή/αποθήκευση αποτελεσμάτων)
        vname = sprintf('%s_rank%d.avi', videos{v}, rnk);
        vwr   = VideoWriter(vname,'Motion JPEG AVI'); open(vwr);
        cellfun(@(f) writeVideo(vwr,f), reconSeq); close(vwr);
    end

    %% 4. Κατά ορμαθούς 100 καρέ
    for rnk = rankList
        Xhat = zeros(size(X));
        for b = 1:batchSz:nFrames
            idx  = b:min(b+batchSz-1,nFrames);
            Xhat(:,idx) = lowrank_reconstruct(X(:,idx), rnk);
        end
        reconSeq = matrix2frames(Xhat, r, c);
        vname = sprintf('%s_rank%d_batch.avi', videos{v}, rnk);
        vwr   = VideoWriter(vname,'Motion JPEG AVI'); open(vwr);
        cellfun(@(f) writeVideo(vwr,f), reconSeq); close(vwr);
    end

    %% 5–7. PCA (με κεντράρισμα)
    Xo   = X - mean(X,2);              % αφαίρεση μέσης τιμής κάθε στήλης
    C    = Xo'*Xo;                     % Τ × Τ
    [V, D] = eig(C,'vector');          % D : ιδιοτιμές, V : ιδιoδιάνυσματα
    [~,ord] = sort(D,'descend'); V = V(:,ord);
    Uhat = Xo*V./sqrt(D(ord)');        % πίνακας ιδιοπροσώπων (Eigenfaces style)

    for rnk = rankList
        Xhat = Uhat(:,1:rnk) * (Uhat(:,1:rnk)' * Xo) + mean(X,2);
        reconSeq = matrix2frames(Xhat, r, c);
        vname = sprintf('%s_PCA_rank%d.avi', videos{v}, rnk);
        vwr   = VideoWriter(vname,'Motion JPEG AVI'); open(vwr);
        cellfun(@(f) writeVideo(vwr,f), reconSeq); close(vwr);
    end
end
