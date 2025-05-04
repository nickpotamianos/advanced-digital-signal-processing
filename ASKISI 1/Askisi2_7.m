% -------------------------------------------------------------------------
%  ΑΣΚΗΣΗ 2.6  (Π-οστή τάξη) –  Επίδραση των M & N στην εκτιμημένη R_xx
%  ------------------------------------------------------------------------
%  • P  : πλήθος αρμονικών συνιστωσών (complex exponentials)
%  • M  : μήκος κάθε υλοποίησης
%  • N  : πλήθος ανεξάρτητων υλοποιήσεων  (snapshot–averaging)
%  • Για κάθε ζεύγος (M,N) υπολογίζονται RMSE σε:
%       – πλάτη |A_i|
%       – συχνότητες ω_i
%       – διασπορά θορύβου σ_w^2
% -------------------------------------------------------------------------

clear; clc; rng('default');                           % αναπαραγωγή
% ------------  ΠΡΑΓΜΑΤΙΚΕΣ ΠΑΡΑΜΕΤΡΟΙ  -----------------------------------
P          = 3;                                       % τάξη διαδικασίας
A_true     = [ 1  sqrt(2)  1.5 ];                    % |A_i|
omega_true = [ 0.20*pi  0.45*pi  0.70*pi ];          % ω_i  (rad)
sigma2_W   = 0.5;                                     % διασπορά θορύβου

% ------------  Λίστες τιμών M & N προς διερεύνηση  -----------------------
M_list = [ 8   12  16  24 ];                          % μήκος υλοπ.
N_list = [ 10  30  60  120  250 ];                    % # υλοποιήσεων

% ------------  Αποθήκευση RMSE  ------------------------------------------
rmse_A  = zeros(numel(M_list),numel(N_list));
rmse_w  = zeros(numel(M_list),numel(N_list));
rmse_s2 = zeros(numel(M_list),numel(N_list));

% ------------  Monte-Carlo επαναλήψεις  ----------------------------------
Lrep = 200;                                           % επαναλήψεις

for iM = 1:numel(M_list)
    M  = M_list(iM);
    n  = (0:M-1).';
    for iN = 1:numel(N_list)
        N  = N_list(iN);

        errA  = zeros(Lrep,P);
        errw  = zeros(Lrep,P);
        errs2 = zeros(Lrep,1);

        for rep = 1:Lrep
            % --------- ΔΗΜΙΟΥΡΓΙΑ N ΥΛΟΠΟΙΗΣΕΩΝ  ------------------------
            X = zeros(N,M);
            for k = 1:N
                phi  = 2*pi*rand(1,P) - pi;           % τυχαίες φάσεις

           
                E      = exp( 1j * ( (0:M-1).' * omega_true ) ); % M×P
                s      = (A_true .* exp(1j*phi)).';              % P×1
                signal = (E * s).';                              % 1×M

                noise  = sqrt(sigma2_W/2)*(randn(1,M)+1j*randn(1,M));
                X(k,:) = signal + noise;
            end

            % --------- ΕΚΤΙΜΗΣΗ μητρώου αυτοσυσχέτισης R̂ --------------
            Rhat = (X'*X)/N;

            % --------- ΙΔΙΟΑΝΑΛΥΣΗ – υποχώροι --------------------------
            [U,lam]   = eig(Rhat,'vector');
            [lam_sorted,idx] = sort(lam,'descend');
            Us = U(:,idx(1:P));                       % υπόχωρος σήματος
            Un = U(:,idx(P+1:end));                   % υπόχωρος θορύβου
            sigma2_hat = mean( lam_sorted(P+1:end) ); % θόρυβος

            % --------- Εκτίμηση συχνοτήτων : MUSIC ----------------------
            % (grid search – αρκεί για αξιολόγηση RMSE)
            omega_grid = linspace(0,pi,4096);
            P_MUSIC    = zeros(size(omega_grid));
            for m = 1:numel(omega_grid)
                a = exp(-1j*omega_grid(m)*(0:M-1)).';
                P_MUSIC(m) = 1 / real( a'*(Un*Un')*a );
            end
            [~,locs]  = findpeaks(P_MUSIC,'SortStr','descend',...
                                  'NPeaks',P);
            omega_hat = sort( omega_grid(locs) );

            % --------- Εκτίμηση ισχύος |A|²  (LS σε υπερκαθορισμένο) ----
            E   = exp(-1j*n*omega_hat);               % M×P
            S   = Rhat - sigma2_hat*eye(M);
            K   = zeros(M^2,P);
            for p = 1:P
                K(:,p) = reshape( E(:,p)*E(:,p)',[],1 );
            end
            a2_hat    = real( pinv(K) * S(:) );       % |A|²
            A_hat     = sqrt(a2_hat).';

            % --------- Σφάλματα επανάληψης ------------------------------
            errA(rep,:)  = (A_hat - A_true).^2;
            errw(rep,:)  = angle( exp(1j*(omega_hat - omega_true)) ).^2;
            errs2(rep)   = (sigma2_hat - sigma2_W)^2;
        end

        % --------- RMSE σε (M,N) ---------------------------------------
        rmse_A (iM,iN)  = sqrt( mean(errA ,'all') );
        rmse_w (iM,iN)  = sqrt( mean(errw ,'all') );
        rmse_s2(iM,iN)  = sqrt( mean(errs2) );
    end
end

% ------------  ΠΙΝΑΚΕΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ------------------------------------
fprintf('\nRMSE |A|   (γραμμές: M, στήλες: N)\n');  disp(rmse_A );
fprintf('\nRMSE ω (rad)   \n');                     disp(rmse_w );
fprintf('\nRMSE σ_w^2     \n');                     disp(rmse_s2);
