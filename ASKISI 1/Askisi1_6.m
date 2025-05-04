% ---------------------------------------------------------------
% ΕΠΙΔΕΙΞΗ του ρόλου των M και N στην εκτίμηση Pisarenko (P = 1)
% ---------------------------------------------------------------

clear; clc; rng('default');           % αναπαραγωγή

% --- Πραγματικές παράμετροι στοχαστικής διαδικασίας ---
A_true   = 2^(3/4);                  % |A|
omega_true = -pi/4;                  % ω
sigma2_W_true = 3 - 2*sqrt(2);       % σ_W^2   (από την άσκηση)

% Λίστες τιμών Μ και N προς διερεύνηση
M_list = [  8   16   32   64 ];      % μήκος υλοποίησης
N_list = [ 10   20   50  100  200 ]; % πλήθος υλοποιήσεων

% Πίνακες αποθήκευσης σφαλμάτων RMSE
RMSE_A  = zeros(length(M_list),length(N_list));
RMSE_w  = zeros(length(M_list),length(N_list));
RMSE_s2 = zeros(length(M_list),length(N_list));

% ----------- ΚΥΡΙΟΣ ΒΡΟΧΟΣ  πάνω σε (M,N) ----------------------
for iM = 1:length(M_list)
    M = M_list(iM);
    
    % προ‑αποθήκευση κατανομών σφαλμάτων για όλες τις επαναλήψεις
    for iN = 1:length(N_list)
        N    = N_list(iN);
        Lrep = 200;                 % # Monte‑Carlo επαναλήψεων
        errA = zeros(Lrep,1);
        errw = zeros(Lrep,1);
        errs2= zeros(Lrep,1);
        
        for rep = 1:Lrep
            % ----------- ΔΗΜΙΟΥΡΓΙΑ N ΥΛΟΠΟΙΗΣΕΩΝ ---------------
            X = zeros(N,M);
            for n = 1:N
                phi  = (2*pi).*rand - pi;            % τυχαία φάση
                k    = 0:M-1;
                signal = A_true * exp( 1j*(omega_true*k + phi) );
                noise  = sqrt(sigma2_W_true/2).*(randn(1,M)+1j*randn(1,M));
                X(n,:) = signal + noise;
            end
            
            % ----------- ΕΚΤΙΜΗΣΗ ΜΗΤΡΩΟΥ ΑΥΤΟΣΥΣΧΕΤΙΣΗΣ ----------
            Rhat = (X'*X)/N;                        % Μ×Μ
            
            % ----------- ΙΔΙΟΑΝΑΛΥΣΗ ------------------------------
            [U,lam] = eig(Rhat,'vector');
            [lambda_max,idxMax] = max(lam);
            lambda_min = min(lam);
            
            % ----------- EKTIMΗΣΕΙΣ ΠΑΡΑΜΕΤΡΩΝ --------------------
            sigma2_hat = lambda_min;                % εκτίμηση θορύβου
            A_hat      = sqrt( (lambda_max - lambda_min)/M );
            u_max      = U(:,idxMax);
            omega_hat  = -angle( u_max(1)/u_max(2) );
            
            % ----------- ΣΦΑΛΜΑΤΑ ---------------------------------
            errA(rep)  = (A_hat      - A_true     )^2;
            errw(rep)  = angle(exp(1j*(omega_hat-omega_true)))^2; % wrap 
            errs2(rep) = (sigma2_hat - sigma2_W_true)^2;
        end
        
        % Μέσο τετραγωνικό σφάλμα (RMSE)
        RMSE_A (iM,iN) = sqrt( mean(errA ) );
        RMSE_w (iM,iN) = sqrt( mean(errw) );
        RMSE_s2(iM,iN) = sqrt( mean(errs2) );
    end
end

% ----------- ΟΠΤΙΚΟΠΟΙΗΣΗ ή ΠΙΝΑΚΕΣ ΑΠΟΤΕΛΕΣΜΑΤΩΝ -------------
fprintf('\n   RMSE πλάτους |A|  (γραμμές: M, στήλες: N)\n'); disp(RMSE_A);
fprintf('\n   RMSE συχνότητας ω (rad)  \n');                  disp(RMSE_w);
fprintf('\n   RMSE διασποράς σ_W^2     \n');                  disp(RMSE_s2);

% Προαιρετικά: σχεδιασμός θερμικών χαρτών σφάλματος
%{
figure; imagesc(N_list,M_list,RMSE_s2); colorbar
xlabel('N'); ylabel('M'); title('RMSE σ^2_W');
%}
