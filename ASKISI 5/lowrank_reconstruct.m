function Xhat = lowrank_reconstruct(X, r)
    % Διατηρεί μόνο τις r μεγαλύτερες ιδιάζουσες τιμές
    [U,S,V]   = svd(X','econ');   % Xᵀ = U Σ Vᵀ  (πιο γρήγορο όταν T≪N)
    S(r+1:end,r+1:end) = 0;
    Xhat = (U*S*V')';            % αντιστρέφουμε τη μεταφορά
end