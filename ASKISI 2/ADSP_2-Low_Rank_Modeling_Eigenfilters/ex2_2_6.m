%% ex6_U_and_W.m
% U(n,θ) και W(n,θ) μέσω ΓΧΑ και επανάληψη ανάλυσης

clear; clc; close all;

%% 0.  Παράμετροι
M         = 1e4;         % μήκος κάθε υλοποίησης
Klist     = [10 100];    % πλήθη υλοποιήσεων προς δοκιμή
sigma2_W  = 0.10;        % ισχύς (διασπορά) του λευκού W
sigma2_Ue = 0.05;        % ισχύς λευκού θορύβου διέγερσης του AR(2)

% AR(2):  H(z)=1/(1 - 0.7 z^{-1} + 0.2 z^{-2})
aU = [1 -0.7 0.2];   bU = 1;   % συντελεστές μεταφοράς

%% 1.  Γένεση πλήρους συνόλου K_max = max(Klist) υλοποιήσεων
Kmax = max(Klist);

% (i) U: φίλτρο AR(2) ← λευκός θόρυβος
Ue_full = sqrt(sigma2_Ue) * randn(M, Kmax);
U_full  = filter(bU, aU, Ue_full);      % [M × Kmax]

% (ii) W: λευκός θόρυβος ισχύος sigma2_W
W_full  = sqrt(sigma2_W) * randn(M, Kmax);

%% 2.  Βρόχος για κάθε K
for K = Klist
    fprintf('\n================  K = %d  ================\n', K);

    % --- 2.1  Υποσύνολα
    U  = U_full(:, 1:K);
    W  = W_full(:, 1:K);

    % --- 2.2  Εκτίμηση C_UU και C_WW
    Uc = U - mean(U, 2);
    Wc = W - mean(W, 2);

    C_UU = (Uc * Uc.') / (K - 1);
    C_WW = (Wc * Wc.') / (K - 1);

    % --- 2.3  Ιδιοανάλυση & ιδιοφίλτρο για U
    [V, D]   = eig(C_UU);
    [lam,ix] = sort(diag(D), 'descend');
    hU       = V(:, ix(1));           % κύριο ιδιοδιάνυσμα
    hU       = hU / norm(hU);         % κανονικοποίηση

    % --- 2.4  Φιλτράρισμα κάθε υλοποίησης
    YU = filter(hU, 1, U);            % [M × K]
    YW = filter(hU, 1, W);            % (ίδιο φίλτρο πάνω σε λευκό)

    % --- 2.5  Εμφάνιση βασικών μεγεθών
    fprintf('  Max eig(C_UU)  : %.4f\n', lam(1));
    fprintf('  Var(off-diag)  C_UU : %.3e\n', ...
            var(C_UU(tril(true(M),-1))));
    fprintf('  Var(off-diag)  C_WW : %.3e\n', ...
            var(C_WW(tril(true(M),-1))));

    % --- 2.6  Ενδεικτικές γραφικές
    figure('Name',sprintf('Heatmaps  K=%d',K));
      subplot(1,2,1); imagesc(C_UU); axis xy; colorbar;
      title(['C_{UU},  K=' num2str(K)]);
      subplot(1,2,2); imagesc(C_WW); axis xy; colorbar;
      title(['C_{WW},  K=' num2str(K)]);

    figure('Name',sprintf('Υλοποιήσεις U και Y_U  (K=%d)',K));
      subplot(2,1,1); plot(U(:,1:min(20,K))); grid on;
      title('Πρώτες 20 υλοποιήσεις  U(n)');
      subplot(2,1,2); plot(YU(:,1:min(20,K))); grid on;
      title('Φιλτραρισμένες  Y_U(n)');

    figure('Name',sprintf('Υλοποιήσεις W και Y_W  (K=%d)',K));
      subplot(2,1,1); plot(W(:,1:min(20,K))); grid on;
      title('Πρώτες 20 υλοποιήσεις  W(n)');
      subplot(2,1,2); plot(YW(:,1:min(20,K))); grid on;
      title('Φιλτραρισμένες  Y_W(n)');
end
