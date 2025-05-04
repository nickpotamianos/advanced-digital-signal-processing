function [D,X] = KSVD(D, X, Y)

for k = 1:size(D,2)

	ind = find(X(k,:) ~= 0);
	
	% COMPUTE ERROR MATRIX
	E = Y - (D*X) + D(:,k) * X(k,:);
	% REDUCE ERROR MATRIC TO THE COLUMNS
	E_r = E(:,ind);

	
	if ~isempty(ind)
		[U,S,V] = svds(E_r, 1, 'largest');
		D(:,k) = U;
		X(k,:) = 0;
		X(k,ind) = S*V';
	end
	
end

end
