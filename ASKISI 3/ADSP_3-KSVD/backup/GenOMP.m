function[x] = GenOMP(D, y, T0, err)


x = zeros(1,100);

% INITIALIZATION
r = y;

% FIRST TIME
[~,k] = max(D' * r);
S = k;
x(k) = D(:,k)'*r;
proj = D(:,S) * pinv((D(:,S)'*D(:,S))) * D(:,S)';
r = (eye(size(proj)) - proj) * y;

% LOOP UNTIL CONDITIONS MET
while (norm(r) > err &&  length(S) <= T0)
	
	[~,k] = max(D' * r);
	S = [S k];
	x(k) = D(:,k)'*r;
	proj = D(:,S) * pinv((D(:,S)'*D(:,S))) * D(:,S)';
	r = (eye(size(proj)) - proj) * y; 

end

end   
