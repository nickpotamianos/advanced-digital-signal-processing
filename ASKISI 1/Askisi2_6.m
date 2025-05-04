%% ------------------------------------------------------------------------
%  ΑΣΚΗΣΗ 2.6  –  Τελικό script
% -------------------------------------------------------------------------
clc; clear; close all;

%% (0) Δοσμένο R_xx  (M = 3,  P = 2)
R = [  6                 1.92705-4.58522j   -3.42705-3.49541j ;
       1.92705+4.58522j   6                 1.92705-4.58522j ;
      -3.42705+3.49541j   1.92705+4.58522j   6               ];
M = size(R,1);                 n = (0:M-1).';

%% (1) Ιδιοανάλυση  →  λευκός θόρυβος
[U,lambda]      = eig(R,'vector');
[sigma2,ix] = min(lambda);          % ιδιοτιμή θορυβου
uN         = U(:,ix);          % ιδιοδιάνυσμα θορύβου

% ▶ Ταξινόμηση ιδιοτιμών σε φθίνουσα σειρά για αναφορά
lambda_sorted = flipud( sort(lambda) );   % [λ1  λ2  λ3]

%% (2) Root–MUSIC  →  συχνότητες
c      = conj(uN).';           % συντελεστές πολυωνύμου
rho    = roots(c);             % 3 ρίζες
rho    = rho( abs(rho) < 1.05 );
omega  = sort( mod(-angle(rho),2*pi) );   % [ω1,ω2]

%% (3) Steering-matrix & «σήμα» S
e  = @(w) exp(-1j*w.*n);       % βοηθ. ανώνυμη συνάρτηση
E  = [ e(omega(1)) , e(omega(2)) ];       % M×P
S  = R - sigma2*eye(M);                       % συνιστώσα σήματος

%% (4) ΠΛΑΤΗ με ελάχιστα τετράγωνα
K  = [ reshape(e(omega(1))*e(omega(1))',[],1), ...
       reshape(e(omega(2))*e(omega(2))',[],1) ];   % (M²)×P
a2 = real( K \ S(:) );           % |A|² λύση LS
A  = sqrt(abs(a2));              % |A|


%% (5) Αποτελέσματα
fprintf('\n---------   Αποτελέσματα Άσκησης 2.6   ---------\n');
fprintf('sigma_w^2    = %.6f\n' , sigma2);
fprintf('ω (rad)      = %.4f   %.4f\n' , omega);
fprintf('|A|          = %.4f   %.4f\n' , A);
% ▶ Εκτύπωση ιδιοτιμών
fprintf('Ιδιοτιμές λ  = %.4f   %.4f   %.4f\n' , lambda_sorted);
fprintf('-----------------------------------------------\n');
%% (6) Pseudospectrum P_inv(e^{jω})
w_plot = linspace(0,2*pi,1024);
P_inv  = 1./abs( polyval(c,exp(1j*w_plot)) ).^2;

figure('Color','w');
plot(w_plot,10*log10(P_inv),'b','LineWidth',1.2); grid on;
xlabel('\omega (rad)'), ylabel('10log_{10}P_{inv}');
title('Root–MUSIC pseudospectrum  (M = 3)');
hold on;

% επισήμανση συχνοτήτων
yl = ylim;  stem(omega, yl(2)*ones(size(omega)), 'r','filled');
legend('P_{inv}','\omega_1, \omega_2','Location','NorthEast');
