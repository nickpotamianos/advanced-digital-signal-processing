%% ex2_low_rank_modeling.m
% ------------------------------------------------------------
% ADSP 2024 – Μοντελοποίηση Χαμηλής Τάξης (Exercise 2–6)
% ------------------------------------------------------------
% Το script εκτελεί ΟΛΑ τα ζητούμενα των Βημάτων 2–6 του
% εγγράφου «Low-Rank Modeling – Eigenfilters» χρησιμοποιώντας
% μεθόδους φιλικές στη μνήμη για πολύ-μεγάλα δεδομένα.
% ------------------------------------------------------------
clear; clc; close all;

%% ---------------- 0. Parameters & options -----------------
SAVE_FIGS   = true;           % σώσε διαγράμματα .png
SAVE_DATA   = true;           % σώσε .mat με αποτελέσματα
PLOT_LINEWD = 1.8;
PLOT_FS     = 12;
noisy_SNRdB = [Inf 40 30 20 10 5 0 -5];  % 1η τιμή = αθόρυβο
rng(0);                       % για αναπαραγωγιμότητα

%% ---------------- 1. Load & basic inspection --------------
load U.mat                     % μεταβλητή U: K×M  (100×100000)
[K,M] = size(U);
fprintf('Loaded U  –  K = %d realizations,   M = %d samples\n',K,M);

%% ---------------- 2. Centering & memory-savvy SVD ---------
ubar   = mean(U,1);            % 1×M  μέση τιμή ανα δείγμα
Uc     = U - ubar;             % κεντραρισμένος πίνακας K×M

% Economy SVD   (Uc = U_s*S_s*V_s')  ➜  U_s:K×K,  S_s:K×K,  V_s:M×K
[U_s,S_s,V_s] = svd(Uc,'econ');
lambda = diag(S_s).^2 ./ (K-1);   % ιδιοτιμές (φθίνουσα)
energy = sum(lambda);

figure;
stem(1:K,lambda,'filled');
title('Ιδιοτιμές του C_{UU}');
xlabel('m'); ylabel('\lambda_m');
set(gca,'FontSize',PLOT_FS);

%% ---------------- 3. Theoretical & empirical verification --------
P_vec    = 1:K;
err_theo = zeros(1,K);
err_emp  = zeros(1,K);

for idx = 1:K
    P = P_vec(idx);
    % Θεωρητικό σφάλμα: Σ_{m=P+1}^K λ_m
    err_theo(idx) = sum(lambda(P+1:end));
    % Πειραματικό σφάλμα: ||Uc - Uc_recon||^2
    Q_P    = V_s(:,1:P);               
    Uc_hat = (Uc * Q_P) * Q_P.';      
    E_emp  = Uc - Uc_hat;              
    err_emp(idx) = mean( sum(E_emp.^2,2) );
end

% Κανονικοποίηση σε σχετικό σφάλμα
relErr_th = err_theo / energy;
relErr_em = err_emp  / energy;

fig0 = figure('Name','Theory vs Empirical Error');
plot(P_vec, 10*log10(relErr_th), '-','LineWidth',PLOT_LINEWD); hold on;
plot(P_vec, 10*log10(relErr_em), '--o','LineWidth',1.2);
grid on;
xlabel('P'); ylabel('10·log_{10}(relative error) [dB]');
title('Άσκηση 3: Θεωρητικό vs Πειραματικό σφάλμα');
legend('Θεωρία','Πείραμα','Location','northeast');
set(gca,'FontSize',PLOT_FS);

if SAVE_FIGS
    saveas(fig0,'compare_err_theory_empirical.png');
end

%% ---------------- 4. Relative error – noiseless channel ----
fig1 = figure('Name','Relative error – noiseless');
plot(P_vec, 10*log10(relErr_th), 'LineWidth',PLOT_LINEWD);
grid on; xlabel('P');
ylabel('10·log_{10}(relative error) [dB]');
title('Σχετικό σφάλμα μετάδοσης σε αθόρυβο κανάλι');
set(gca,'FontSize',PLOT_FS);

if SAVE_FIGS
    saveas(fig1,'relErr_noiseless.png');
end

%% ---------------- 5. Noisy channel simulation (single P_opt) -----
P_opt        = 25;   % επιλεγμένο P για μελέτη
relErr_noisy = zeros(length(noisy_SNRdB),1);

for ii = 1:length(noisy_SNRdB)
    SNRdB = noisy_SNRdB(ii);
    if isinf(SNRdB)
        sigma2 = 0;
    else
        sigma2 = energy / (M * 10^(SNRdB/10));
    end

    % Ανακατασκευή με P_opt
    Q_P   = V_s(:,1:P_opt);
    W     = sqrt(sigma2) * randn(size(U));
    Xc    = (U + W) - ubar;
    Y_hat = (Xc * Q_P) * Q_P.';
    E_hat = Y_hat - Uc;
    relErr_noisy(ii) = mean( sum(E_hat.^2,2) ) / energy;
end

fig2 = figure('Name','Relative error vs SNR');
plot(noisy_SNRdB, 10*log10(relErr_noisy), '-o', 'LineWidth',PLOT_LINEWD);
grid on; xlabel('SNR [dB]');
ylabel('10·log_{10}(relative error) [dB]');
title(sprintf('Σφάλμα για P = %d σε ενθόρυβο κανάλι',P_opt));
set(gca,'FontSize',PLOT_FS);

if SAVE_FIGS
    saveas(fig2,'relErr_vs_SNR.png');
end

%% ---------------- 6. Additional experiments: various P_opts -------
P_opts = [10 20 25 30 40];
fig3   = figure('Name','Σφάλμα vs SNR για διάφορα P');
hold on;
for pp = 1:length(P_opts)
    Pp = P_opts(pp);
    relErr_tmp = zeros(length(noisy_SNRdB),1);
    for ii = 1:length(noisy_SNRdB)
        SNRdB = noisy_SNRdB(ii);
        if isinf(SNRdB)
            sigma2 = 0;
        else
            sigma2 = energy / (M * 10^(SNRdB/10));
        end
        Q_P   = V_s(:,1:Pp);
        Xc    = (U + sqrt(sigma2)*randn(size(U))) - ubar;
        Y_hat = (Xc * Q_P) * Q_P.';
        E_hat = Y_hat - Uc;
        relErr_tmp(ii) = mean( sum(E_hat.^2,2) ) / energy;
    end
    plot(noisy_SNRdB,10*log10(relErr_tmp),'-o','DisplayName',sprintf('P=%d',Pp),'LineWidth',1.2);
end
grid on;
xlabel('SNR [dB]'); ylabel('10·log_{10}(relative error) [dB]');
title('Σφάλμα vs SNR για διάφορα P');
legend('Location','southwest');
set(gca,'FontSize',PLOT_FS);

if SAVE_FIGS
    saveas(fig3,'relErr_vs_SNR_various_P.png');
end

%% ---------------- 7. Save data --------------------------
if SAVE_DATA
    save('ex2_results.mat', 'P_vec','relErr_th','relErr_em', ...
         'noisy_SNRdB','relErr_noisy','P_opts','lambda');
end

fprintf('\nΟλοκληρώθηκε επιτυχώς η προσομοίωση όλων των βημάτων.\n');
