%% Βήμα 3 – Γραφικές παραστάσεις
% -------------------------------------------------------------
% προϋποθέτει:  X  [M×K]  |  Y  [M×K]  |  n [M×1]  |  K (# υλοποιήσεων)
% -------------------------------------------------------------

%% 3α) Εκτίμηση του μητρώου C_xx
Xc   = X - mean(X, 2);           % κεντράρισμα
C_xx = (Xc * Xc.') / (K - 1);    % [M×M] εκτιμητής συνδιασποράς

figure('Name','C_{xx}');
subplot(1,2,1);
imagesc(C_xx);  axis xy;  colorbar;
title('imagesc(C_{xx})','Interpreter','latex');

subplot(1,2,2);
mesh(C_xx);
title('mesh(C_{xx})','Interpreter','latex');
xlabel('n'); ylabel('m'); zlabel('C_{xx}(n,m)');

%% 3β) Όλες οι υλοποιήσεις της διαδικασίας X(n)
figure('Name','Υλοποιήσεις X');
plot(n, X, 'LineWidth', 0.8);
xlabel('n'); ylabel('X(n;k)'); grid on;
title('Όλες οι υλοποιήσεις X(n;\varphi_k)','Interpreter','latex');

%% 3γ) Μέση διαδικασία X̄(n)
X_mean = mean(X, 2);
figure('Name','Μέση X');
stem(n, X_mean, 'filled');
xlabel('n'); ylabel('\bar{X}(n)'); grid on;
title('Μέση διαδικασία \bar{X}(n)','Interpreter','latex');

%% 3δ) Όλες οι υλοποιήσεις της αποθορυβοποιημένης (φιλτραρισμένης) διαδικασίας Y(n)
figure('Name','Υλοποιήσεις Y');
plot(n, Y, 'LineWidth', 0.8);
xlabel('n'); ylabel('Y(n;k)'); grid on;
title('Υλοποιήσεις της αποθορυβοποιημένης διαδικασίας Y(n)','Interpreter','latex');

%% 3ε) Μέση αποθορυβοποιημένη διαδικασία  Ȳ(n)
Y_mean = mean(Y, 2);
figure('Name','Μέση Y');
stem(n, Y_mean, 'filled');
xlabel('n'); ylabel('\bar{Y}(n)'); grid on;
title('Μέση διαδικασία \bar{Y}(n)','Interpreter','latex');
