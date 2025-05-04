% ------------------------------------------------------------------------
%  Askisi2_8  –  P = 5,  M = 50,  N = 100   (Root-MUSIC & EV)
% ------------------------------------------------------------------------
clear;  clc;  rng('default')

% === 1. Πραγματικές παράμετροι ==========================================
P            = 5;
A_true       = [1  sqrt(2)  1.5  1  1];            % |A_i|
omega_set{1} = [.20 .40 .50 .75 .88]*pi;           % 8-α
omega_set{2} = [.18 .34 .50 .77 .91]*pi;           % 8-β (σετ 2)
omega_set{3} = [.10 .35 .60 .70 .92]*pi;           % 8-β (σετ 3)
sigma2_W     = 0.75;

M = 50;  N = 100;          % μήκος & πλήθος υλοποιήσεων
K = numel(omega_set);      % πόσα διαφορετικά σετ συχνοτήτων

methods = {'RootMUSIC','EV'};  nMeth = 2;
RMSE_w = zeros(K,nMeth);  RMSE_A = RMSE_w;  RMSE_s2 = RMSE_w;

n = (0:M-1).';                                    % 50×1

% === 2. Βρόχος επάνω στα K σετ συχνοτήτων ===============================
for ks = 1:K
    omega_true = omega_set{ks};

    % --- δημιουργία N υλοποιήσεων --------------------------------------
    E_true = exp(1j*n*omega_true);                % 50×5 (σταθερό)
    X = zeros(N,M);
    for k = 1:N
        phi   = 2*pi*rand(1,P)-pi;
        s_vec = (A_true .* exp(1j*phi)).';        % 5×1
        X(k,:)= (E_true*s_vec).'+ ...
                sqrt(sigma2_W/2)*(randn(1,M)+1j*randn(1,M));
    end

    % --- εκτιμώμενο R̂ και ιδιοανάλυση ---------------------------------
    Rhat = (X'*X)/N;
    [U,lam] = eig(Rhat,'vector');
    [lam_s,idx] = sort(lam,'descend'); U = U(:,idx);
    sigma2_hat = mean(lam_s(P+1:end));
    Un = U(:,P+1:end);                            % υπόχωρος θορύβου
    S  = Rhat - sigma2_hat*eye(M);

    % --- κοινό πλέγμα MUSIC/EV ----------------------------------------
    omega_grid = linspace(0,pi,4096);

    %% ===== Root-MUSIC ==================================================
    P_M = zeros(size(omega_grid));
    for m = 1:numel(omega_grid)
        a = exp(-1j*omega_grid(m)*(0:M-1)).';
        P_M(m) = 1 / real( a'*(Un*Un')*a );
    end
    [~,ord] = sort(P_M,'descend');
    peaks = [];                                    % κορυφές με «απόσταση»
    
    % FIX: Ensure ord is a column vector and iterate through its elements
    ord = ord(:);  % Make sure ord is a column vector
    for i = 1:length(ord)
        idx = ord(i);
        if isempty(peaks) || all(abs(idx-peaks) > 5)
            peaks(end+1) = idx;
        end
        if numel(peaks)==P, break, end
    end
    
    music_w = reshape(sort(omega_grid(peaks)),1,P);  % 1×P
    E_mus   = exp(-1j*n*music_w);                  % 50×5
    a2_mus  = real( pinv(E_mus.'*E_mus) * (E_mus.'*S*E_mus) );
    A_mus   = sqrt(max(a2_mus,0)).';

    %% ===== EV (EigenVector) ===========================================
    wEV   = 1 ./ lam_s(P+1:end);
    P_EV  = zeros(size(omega_grid));
    for m = 1:numel(omega_grid)
        a = exp(-1j*omega_grid(m)*(0:M-1)).';
        P_EV(m) = 1 / real( a'*(Un*diag(wEV)*Un')*a );
    end
    [~,ord] = sort(P_EV,'descend');
    peaks = [];
    
    % FIX: Same fix for the EV method
    ord = ord(:);  % Make sure ord is a column vector
    for i = 1:length(ord)
        idx = ord(i);
        if isempty(peaks) || all(abs(idx-peaks) > 5)
            peaks(end+1) = idx;
        end
        if numel(peaks)==P, break, end
    end
    
    ev_w  = reshape(sort(omega_grid(peaks)),1,P);
    E_ev  = exp(-1j*n*ev_w);
    a2_ev = real( pinv(E_ev.'*E_ev) * (E_ev.'*S*E_ev) );
    A_ev  = sqrt(max(a2_ev,0)).';

    %% ===== υπολογισμός RMSE ===========================================
    est_w = [music_w ; ev_w];
    est_A = [A_mus  ; A_ev ];
    for mth = 1:nMeth
        RMSE_w(ks,mth)=sqrt(mean(angle(exp(1j*(est_w(mth,:)-omega_true))).^2));
        RMSE_A(ks,mth)=sqrt(mean((abs(est_A(mth,:))-A_true).^2));
        RMSE_s2(ks,mth)=abs(sigma2_hat - sigma2_W);
    end
end

% === 3. Αποτελέσματα 8-α ===============================================
disp('=== RMSE (Σετ 1) ===')
T1 = table(methods.',RMSE_w(1,:).',RMSE_A(1,:).',RMSE_s2(1,:).',...
           'VariableNames',{'Μέθοδος','RMSE_ω','RMSE_|A|','RMSE_σ2'});
disp(T1)

% === 4. Συγκεντρωτικός πίνακας για 8-β (αν K>1) ========================
if K>1
    fprintf('\n--- RMSE_ω για όλα τα σετ (γραμμές K, στήλες μέθοδοι) ---\n')
    disp(RMSE_w)
end