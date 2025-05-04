function [MSE,D,X] = DictionaryLearning(D, Y, err, T0, numEpochs, X)

MSE = zeros(1,numEpochs);
for epoch = 1:numEpochs
	
	disp(['epoch ', num2str(epoch)])
	parfor i = 1:(size(Y,2))
		X(:,i) = GenOMP(D, Y(:,i), T0, err);
	end

	[D,X] = KSVD(D, X, Y);  
	MSE(epoch) = norm((Y - D*X))^2;
	disp(['mse ', num2str(mean(MSE(:, epoch)))])
	
end

semilogy(MSE)
ylabel('$\log_{10} (\|Y-DX\|_F^2)$','Interpreter','latex','FontSize',14)
xlabel('epoch','Interpreter','latex','FontSize',14)

end