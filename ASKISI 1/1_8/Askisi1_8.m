% ==============================================================
%      ΑΣΚΗΣΗ 8  –  Επαναληπτική ανάλυση ως προς σ^2_W
%      (Πλήρης εφαρμογή 7.1‑7.8 για κάθε M και σ^2_W)
% ==============================================================
clear; clc; close all; rng('default');     % αναπαραγωγιμότητα

% ---------  Σταθερές παραμέτρων σήματος / δειγματοληψίας ----------
N          = 100;                    % αριθμός υλοποιήσεων
M_tot      = 50;                     % μήκος κάθε υλοποίησης
M_set      = [ 2 10 20 30 40 50 ];   % διαστάσεις υπο‑μητρών
A_true     = 3*sqrt(2);              % πλάτος   |A|
omega_true =  pi/5;                  % συχνότητα ω (0.2π rad/δείγμα)

% ---------  Φάσμα εξεταζόμενων τιμών διασποράς θορύβου ----------
sigma2_set = [ 0.05 0.1 0.25 0.5 1 2 ];   % μπορείτε να προσθέσετε επιπλέον

% ---------  Πίνακας συγκεντρωτικών δεικτών ---------------------
results = [];                        % πίνακας [σ2, SNRdB, errA, errw_P, errw_M]

% --------------------------------------------------------------
%     ΕΞΩΤΕΡΙΚΟΣ ΒΡΟΧΟΣ  –  Διαδοχικές τιμές θορύβου
% --------------------------------------------------------------
for sigma2_W = sigma2_set
    
    % ----- Δημιουργία Ν υλοποιήσεων μήκους M_tot ---------------
    X = zeros(N,M_tot);
    for n = 1:N
        phi = (2*pi)*rand - pi;                     % τυχαία φάση
        k   = 0:M_tot-1;
        s   = A_true * exp( 1j*(omega_true*k + phi) );
        w   = sqrt(sigma2_W/2) * ...
              (randn(1,M_tot)+1j*randn(1,M_tot));   % λευκός Γκαουσιανός
        X(n,:) = s + w;
    end
    
    % ----- Δειγματικό μητρώο αυτοσυσχέτισης 50×50 --------------
    Rhat50 = (X'*X)/N;
    
    % ----- Πίνακας τιμών σφάλματος (για Μ=50) ------------------
    errA  = NaN;  errw_P = NaN;  errw_M = NaN;      % θα ενημερωθούν στο M=50
    
    % -----------------------------------------------------------
    %      ΕΣΩΤΕΡΙΚΟΣ ΒΡΟΧΟΣ  –  Διαδοχικά M για 7.1‑7.8
    % -----------------------------------------------------------
    for M = M_set
        R_M = Rhat50(1:M,1:M);            % υπο‑μητρώο R_XX,M
        
        % 7.1 Ιδιοανάλυση
        [U,lambda] = eig(R_M,'vector');
        [lambda,idx] = sort(lambda,'descend'); U = U(:,idx);
        
        % 7.2 Εκτίμηση διασποράς θορύβου
        sigma2_est = mean(lambda(2:end));
        
        % Ιστόγραμμα ιδιοτιμών
        figure('visible','off');
            histogram(lambda,round(sqrt(M)));
            title(sprintf('Histogram λ  (M=%d,  σ^2_W=%.2g)',M,sigma2_W));
            xlabel('\lambda'); grid on;
        saveas(gcf,sprintf('hist_M%d_s%.2g.png',M,sigma2_W));
        
        % 7.3 Εκτίμηση πλάτους
        A_est = sqrt( (lambda(1)-sigma2_est)/M );
        
        % 7.4 Συντελεστές πολυωνύμου (m = 2)
        c = flipud(conj(U(:,2)));         % m = 2
        % 7.5 Εκτίμηση συχνότητας με ρίζες
        z = roots(c);
        z = z( abs(z)>0.95 & abs(z)<1.05 );     % ρίζες κοντά στο |z|=1
        if numel(z)>1
            [~,ctr]=kmeans([real(z),imag(z)],1,'distance','sqeuclidean',...
                            'replicates',5);
            z0 = ctr(1)+1j*ctr(2);
        else
            z0 = z;
        end
        omega_est = -angle(z0);
        
        % 7.6 Q(M,m) (m=2)
        wPlot = linspace(-pi,pi,2048);
        Q_Mm  = 1./abs( polyval(c,exp(-1j*wPlot)).^2 );
        figure('visible','off');
            plot(wPlot/pi,10*log10(Q_Mm)); axis tight;
            title(sprintf('Q (M=%d,m=2)  σ^2_W=%.2g',M,sigma2_W));
            xlabel('\omega/\pi'); ylabel('dB');
        saveas(gcf,sprintf('Q_M%d_s%.2g.png',M,sigma2_W));
        
        % 7.7 MUSIC
        denom = zeros(size(wPlot));
        for m = 2:M
            cm = flipud(conj(U(:,m)));
            denom = denom + abs(polyval(cm,exp(-1j*wPlot))).^2;
        end
        Q_MUSIC = 1./denom;
        figure('visible','off');
            plot(wPlot/pi,10*log10(Q_MUSIC)); axis tight;
            title(sprintf('MUSIC  (M=%d)  σ^2_W=%.2g',M,sigma2_W));
            xlabel('\omega/\pi'); ylabel('dB');
        saveas(gcf,sprintf('MUSIC_M%d_s%.2g.png',M,sigma2_W));
        [~,kmax] = max(Q_MUSIC);
        omega_est_MUSIC = wPlot(kmax);
        
        % 7.8 EV
        denomEV = zeros(size(wPlot));
        for m = 2:M
            cm = flipud(conj(U(:,m)));
            denomEV = denomEV + (1/lambda(m))*abs(polyval(cm,exp(-1j*wPlot))).^2;
        end
        Q_EV = 1./denomEV;
        figure('visible','off');
            plot(wPlot/pi,10*log10(Q_EV)); axis tight;
            title(sprintf('EV  (M=%d)  σ^2_W=%.2g',M,sigma2_W));
            xlabel('\omega/\pi'); ylabel('dB');
        saveas(gcf,sprintf('EV_M%d_s%.2g.png',M,sigma2_W));
        
        % ---- Δείκτες σφάλματος μόνο για M_tot=50 ---------------
        if M == M_tot
            errA  = abs(A_est-A_true)/A_true;
            errw_P = abs( angle(exp(1j*(omega_est     -omega_true))) )/pi;
            errw_M = abs( angle(exp(1j*(omega_est_MUSIC-omega_true))) )/pi;
        end
    end  % loop M
    
    % ----- Συγκεντρωτικός πίνακας για M=50 ----------------------
    SNRdB = 20*log10( A_true / sqrt(2*sigma2_W) );
    success = errw_M < 0.01;             % επιτυχία MUSIC
    results = [results ; sigma2_W SNRdB errA errw_P errw_M success];
    
end  % loop σ2

% ----------- Εμφάνιση τελικού πίνακα ---------------------------
T = array2table(results,...
     'VariableNames',{'sigma2_W','SNR_dB','relErrA', ...
                      'relErrω_Pis','relErrω_MUSIC','successMUSIC'});
disp(T);

disp('--- Όλες οι εικόνες αποθηκεύτηκαν στον τρέχοντα φάκελο. ---');
