%%
clear
close all
clc
train_set = dir('./train/*.jpg');

path = fullfile(train_set(1).folder, train_set(1).name);
im = im2double(rgb2gray(imread(path)));

% SMALL SCALE VERSION (make sure {W,H} mod patch_size == 0)
% im = imresize(im, [1920/8,1088/8]);

[H, W] = size(im);


% CONSTRUCT INPUT Y
patch_size = 8;
num_col_Y = (H/patch_size)*(W/patch_size)*length(train_set);
Y = zeros(patch_size^2, num_col_Y);



for i = 1:length(train_set)
	path = fullfile(train_set(1).folder, train_set(1).name)
	im = im2double(rgb2gray(imread(path)));
	% SPLIT IMAGE IN PATCHES OF SIZE (patch_size, patch_size)
	patch_cnt = 1;
	for patch_row = 1:patch_size:H
		for patch_column = 1:patch_size:W
			Y(:, patch_cnt) = reshape(im(patch_row:patch_row+patch_size-1, patch_column:patch_column+patch_size-1), [patch_size^2,1]);
			patch_cnt = patch_cnt + 1;
		end
	end
end

% CONSTRUCT INPUT X
X = randn(100, size(Y,2));

% CONSTRUCT INPUT D (NUMBER OF ATOMS SOULD BE LARGER THAN ATOM SIZE(=patch_size))
numAtoms = 100;
D = randn(size(Y,1), numAtoms);

% TOLERATED ERROR |Y-DX|_{F}
err = 1e-4;

% NUMBER OF EPOCHS TO TRAIN THE DICTIONARY FOR
numEpochs = 200;

% SPARSITY INDEX (KEEP T0 NON-ZERO ELEMENTS IN EACH DICTIONARY ATOM)
T0 = 20;

[MSE, new_D, X] = DictionaryLearning(D, Y, err, T0, numEpochs, X);

atoms = {reshape(n_D(:,1),[8,8])};
for i = 2:size(n_D,2)
	atoms = {atoms reshape(n_D(:,i),[8,8])};
	imagesc(atoms(end));
	pause
end
% montage(atoms, 'Size', [10,10]);