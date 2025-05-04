
%% FIR Ιδιοφίλτρο (Eigenfilter) για Αποθορυβοποίηση
clear; close all; clc;

%% Παράμετροι γεννήτριας
N     = 10000;             % Μήκος κάθε υλοποίησης
K_all = [10, 50, 100, 1000];% Τιμές K για μελέτη σύγκλισης
sigmaW = sqrt(0.1);        % STD του λευκού θορύβου

%% Βοηθητικές μεταβλητές για αποθήκευση αποτελεσμάτων
meanX_store = cell(length(K_all),1);
meanY_store = cell(length(K_all),1);
Cxx_store   = cell(length(K_all),1);

%% Κύριος βρόχος για κάθε K
for idxK = 1:length(K_all)
    K = K_all(idxK);

    % Δημιουργία πίνακα υλοποιήσεων X(k,n)
    X = zeros(K, N);
    for k = 1:K
        phi   = 2*pi*rand;                     % Τυχαία φάση
        n     = 0:N-1;
        sinus = sin(2*pi*n/1000 + phi);        % Ημιτονοειδής συνιστώσα
        noise = sigmaW * randn(1, N);          % Λευκός γκαουσιανός θόρυβος
        X(k,:) = sinus + noise;
    end

    % Κεντροποίηση για ακριβή εκτίμηση συνδιασποράς
    X = X - mean(X, 2);  % αφαιρώ τη μέση τιμή κάθε υλοποίησης

    % Εκτίμηση μητρώου συνδιασποράς Cxx
    Cxx = (1/K) * (X.' * X);
    Cxx_store{idxK} = Cxx;

    % Υπολογισμός ιδιοτιμών–ιδιοδιανυσμάτων
    [Q, D] = eig(Cxx);
    [~, imax] = max(diag(D));    % Δείκτης μέγιστης ιδιοτιμής
    h = Q(:, imax);
    h = h / norm(h);             % Κανονικοποίηση ||h||=1

    % Εφαρμογή FIR–φίλτρου σε κάθε υλοποίηση
    Y = filter(h, 1, X.').';     % Φιλτράρισμα κατά μήκος n

    % Υπολογισμός μέσων καμπυλών
    meanX_store{idxK} = mean(X, 1);
    meanY_store{idxK} = mean(Y, 1);

    %% Απεικονίσεις για την τρέχουσα τιμή του K
    figure('Name', sprintf('K = %d',K), 'NumberTitle','off', ...
           'Position',[100 100 1000 800]);

    subplot(3,2,1);
    imagesc(Cxx); colorbar;
    title('Εκτίμηση \bfC_{XX}'); xlabel('Δείκτης n'); ylabel('Δείκτης m');

    subplot(3,2,2);
    plot(X.'); grid on;
    title('Όλες οι υλοποιήσεις X(n)'); xlabel('n'); ylabel('X_k(n)');

    subplot(3,2,3);
    plot(meanX_store{idxK}, 'LineWidth',1.5); grid on;
    title('Μέση \bfX(n)'); xlabel('n'); ylabel('\barX(n)');

    subplot(3,2,4);
    plot(Y.'); grid on;
    title('Όλες οι εξόδοι Y(n)'); xlabel('n'); ylabel('Y_k(n)');

    subplot(3,2,5);
    plot(meanY_store{idxK}, 'LineWidth',1.5); grid on;
    title('Μέση \bfY(n)'); xlabel('n'); ylabel('\barY(n)');

    % Φασματική πυκνότητα μέσης εισόδου
    subplot(3,2,6);
    psd_est = abs(fft(meanX_store{idxK})).^2;
    plot(10*log10(psd_est(1:N/2))); grid on;
    title('PSD(\barX) σε dB'); xlabel('Συχνότητα'); ylabel('Επίπεδο [dB]');
end

%% Παραγωγή ασθενώς στάσιμης διαδικασίας μέσω LTI συστήματος
% Επιλογή συντελεστών FIR για U(n)
b = [0.5, 0.2, 0.3];  % Κρουστική απόκριση LTI

for idxK = 1:length(K_all)
    K = K_all(idxK);

    % Γεννήτρια λευκού θορύβου
    V = randn(K, N);

    % Φιλτράρισμα μέσω LTI για παραγωγή U(n)
    U = filter(b, 1, V.').';

    % Πρόσθεση λευκού θορύβου W
    W  = sigmaW * randn(K, N);
    X2 = U + W;

    % Κεντροποίηση
    X2 = X2 - mean(X2, 2);

    % Εκτίμηση νέου μητρώου συνδιασποράς
    Cxx2 = (1/K) * (X2.' * X2);

    % Υπολογισμός νέου eigenfilter
    [Q2, D2] = eig(Cxx2);
    [~, imax2] = max(diag(D2));
    h2 = Q2(:, imax2);
    h2 = h2 / norm(h2);

    % Εφαρμογή του νέου φίλτρου
    Y2 = filter(h2, 1, X2.').';

    % Απεικόνιση σύγκρισης μέσων πριν/μετά
    figure('Name', sprintf('LTI Process, K = %d',K), ...
           'NumberTitle','off','Position',[150 150 800 400]);
    plot(mean(X2,1), '--', 'LineWidth',1); hold on;
    plot(mean(Y2,1), '-',  'LineWidth',1.5); grid on;
    title(sprintf('Μέσες πριν/μετά eigenfilter (K = %d)',K));
    legend('\barX_2(n)','\barY_2(n)','Location','best');
    xlabel('n'); ylabel('Πλάτος');
end
