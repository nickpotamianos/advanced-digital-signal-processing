% ==============================================================
%  Ερώτημα 7  (ADSP) – Πλήρης υλοποίηση 7.1‑7.8
% ==============================================================
clear; clc; close all; rng('default');          % σταθεροποίηση τυχαιότητας

% ----------  Παράμετροι στοχαστικής διαδικασίας ----------------
N          = 100;           % πλήθος υλοποιήσεων
M_tot      = 50;            % μήκος κάθε υλοποίησης
A_true     = 3*sqrt(2);     % πλάτος   (3√2)
omega_true =  pi/5;         % συχνότητα (0.2π)
sigma2_W   = 0.5;           % διασπορά λευκού θορύβου
M_set      = [ 2 10 20 30 40 50 ];   % ζητούμενα Μ

% ----------  Βήμα 7: Δημιουργία υλοποιήσεων --------------------
X = zeros(N, M_tot);
for n = 1:N
    phi = (2*pi).*rand - pi;              % φ ~ U[-π,π]
    k   = 0:M_tot-1;
    s   = A_true * exp( 1j*(omega_true .* k + phi) );
    w   = sqrt(sigma2_W/2) * (randn(1,M_tot) + 1j*randn(1,M_tot));
    X(n,:) = s + w;
end

% ----------  7α: Στοχαστική μέση τιμή --------------------------
x_bar = mean( X , 1 );                     % 1×50 διάνυσμα
fprintf('\nΜέση τιμή (πλάτος μέτρου) = %.3e (πρέπει ≈ 0)\n',...
        norm(x_bar)/sqrt(M_tot));

% ----------  7β: Μητρώο αυτοσυσχέτισης 50×50 ------------------
Rhat50 = (X' * X) / N;                    % μέθοδος στιγμών

% Αποθήκευση για μελλοντική χρήση (προαιρετικά)
% save Rhat50.mat Rhat50

% =================================================================
%               Βρόχος για κάθε επιμέρους M
% =================================================================
for M = M_set
    R_M = Rhat50(1:M , 1:M);             % MxM υπο‑μητρώο
    
    % ------- 7.1  Ιδιοτιμές / ιδιοδιανύσματα --------------------
    [U,lambda] = eig(R_M,'vector');
    [lambda_sorted, idx] = sort(lambda,'descend');
    U = U(:,idx);                         % ταξινόμηση ίδια με ιδιοτιμές
    
    % ------- 7.2  Εκτίμηση διασποράς θορύβου -------------------
    % Θεωρούμε P = 1 → θόρυβος σε ιδιοτιμές 2:Μ
    sigma2_est = mean( lambda_sorted(2:end) );
    
    % Ιστόγραμμα ιδιοτιμών + κεντρικές ροπές
    figure('visible','off');
       histogram( lambda_sorted , round(sqrt(M)) );
       title(sprintf('Histogram λ (M=%d)',M));
       xlabel('\lambda'); grid on;
    saveas(gcf, sprintf('hist_lambda_M%d.png',M));
    mu1 = mean(lambda_sorted);            % 1ης τάξης κεντρική ροπή = 0
    mu2 = var(lambda_sorted);             % 2ης κεντρική ροπή
    fprintf('M=%2d  --->  σ²_W εκτ.=%.4f   (αληθές=0.5)\n',...
            M, sigma2_est);
    
    % ------- 7.3  Εκτίμηση πλάτους σήματος ----------------------
    A_est = sqrt( (lambda_sorted(1) - sigma2_est)/M );
    
    % ------- 7.4  Τριγωνομετρικά πολυώνυμα ----------------------
    % P(M,m)(e^{jω}) = e_M^H(ω) n_m   για m=2:Μ
    %  -- e_M(ω) = [1  e^{-jω} ... e^{-j(M-1)ω}]^T
    m_idx = 2:M;
    % Παράδειγμα υπολογισμού πολυωνύμου για m=2 (γενικεύεται στο 7.5‑7.8)
    c = flipud(conj(U(:,m_idx(1))));      % συντελεστές πολυωνύμου
    % P(ω) = Σ_{k=0}^{M-1} c_k e^{-j ω k}
    
    % ------- 7.5  Εκτίμηση συχνότητας με ρίζες ------------------
    % Φόρμα πολυωνύμου σε z = e^{-jω}  → συντελεστές c_k
    roots_poly = roots(c);
    % Επιλογή ρίζας στο μοναδιαίο κύκλο |z|≈1 μέσω kmeans
    z_candidates = roots_poly( abs(roots_poly) > 0.95 & abs(roots_poly) < 1.05 );
    if numel(z_candidates) > 1
        [~,C] = kmeans([real(z_candidates), imag(z_candidates)],1,...
                       'Distance','sqeuclidean','Replicates',5);
        z0 = C(1) + 1j*C(2);
    else
        z0 = z_candidates(1);
    end
    omega_est = -angle(z0);               % rad/δείγμα
    
    % ------- 7.6  Q(M,m)(e^{jω}) συνάρτηση ----------------------
    wAxis = linspace(-pi,pi,2048);
    Q_Mm  = 1 ./ abs( polyval(c,wAxis.*0 + exp(-1j*wAxis)).^2 );
    figure('visible','off');
       plot(wAxis/pi, 10*log10(Q_Mm));  axis tight
       title(sprintf('Q(M=%d,m=2) (dB)',M)); xlabel('\omega/\pi');
    saveas(gcf, sprintf('Q_M%d_m2.png',M));
    
    % ------- 7.7  QMUSIC_M(ω) ----------------------------------
    denom = zeros(size(wAxis));
    for m = 2:M
        c_m = flipud(conj(U(:,m)));
        denom = denom + abs( polyval(c_m, exp(-1j*wAxis)).^2 );
    end
    Qmusic = 1 ./ denom;
    figure('visible','off');
       plot(wAxis/pi, 10*log10(Qmusic)); axis tight
       title(sprintf('MUSIC  Q_M (M=%d)',M)); xlabel('\omega/\pi');
    saveas(gcf, sprintf('MUSIC_M%d.png',M));
    
    % ------- 7.8  QEV_M(ω) -------------------------------------
    denomEV = zeros(size(wAxis));
    for m = 2:M
        c_m = flipud(conj(U(:,m)));
        denomEV = denomEV + (1/lambda_sorted(m)) .* ...
                  abs( polyval(c_m, exp(-1j*wAxis)).^2 );
    end
    QEV = 1 ./ denomEV;
    figure('visible','off');
       plot(wAxis/pi, 10*log10(QEV)); axis tight
       title(sprintf('EV  Q_M (M=%d)',M)); xlabel('\omega/\pi');
    saveas(gcf, sprintf('EV_M%d.png',M));
    
    % -------  Αναφορά αποτελεσμάτων ----------------------------
    fprintf('        |A| εκτ.=%.4f   (αληθές=%.4f)\n',A_est, A_true);
    fprintf('        ω  εκτ.=%.4fπ   (αληθές=%.4fπ)\n',...
            omega_est/pi, omega_true/pi);
    fprintf('        μ₂ ιδιοτιμών  = %.4e\n\n', mu2);
end

disp('--- Όλες οι εικόνες (ιστογράμματα / καμπύλες) αποθηκεύτηκαν στον φάκελο εργασίας. ---');
