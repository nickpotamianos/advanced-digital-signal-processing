%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DENOISING/INPAINTING SCRIPT                                        %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script implements denoising and inpainting using a pre-trained 
% dictionary, evaluating performance at different SNR levels.
%=========================================================================

clear; close all; clc;

%% ------------------ PARAMETERS ----------------------------------------
blockSize = 8;        % patch dimension (pixels)
T0        = 8;        % sparsity per block
epsilon   = 0;        % residual threshold (0 ➜ off)
gOMP_N    = 1;        % =1 → OMP, >1 → GenOMP batch size
SNR_levels = [0, 20, 50, 100]; % SNR levels in dB
rng('default');       % reproducibility seed

%% ------------------ LOAD PRETRAINED DICTIONARY ------------------------
load('D.mat');  % Dictionary from previous exercise
fprintf('Loaded pre-trained dictionary with %d atoms.\n', size(D,2));

%% ------------------ DATASET PATHS ------------------------------------
testDir  = fullfile('dataset','test');
imgExt   = {'*.png','*.jpg','*.bmp'};

teFiles = struct([]);
for e = 1:numel(imgExt)
    teFiles = [teFiles; dir(fullfile(testDir,imgExt{e}))];
end
assert(~isempty(teFiles),'No test images found.');
fprintf('Found %d test images.\n',numel(teFiles));

%% ------------------ HELPERS ------------------------------------------
extractBlocks = @(I) im2col(I,[blockSize blockSize],'distinct');
assembleImage = @(B,imSize) col2im(B,[blockSize blockSize],imSize,'distinct');

%% ------------------ DENOISING EXPERIMENT -----------------------------
fprintf('Running denoising experiment...\n');

% Storage for results
numImages = numel(teFiles);
mseDenoising = zeros(numImages, length(SNR_levels));
psnrDenoising = zeros(numImages, length(SNR_levels));

% For each test image
for i = 1:numImages
    % Load and convert to grayscale if needed
    I_orig = im2double(imread(fullfile(testDir,teFiles(i).name)));
    if size(I_orig,3)==3, I_orig = rgb2gray(I_orig); end
    [h, w] = size(I_orig);
    
    % Create a figure for this image
    fig = figure('Name', sprintf('Denoising - %s', teFiles(i).name));
    
    % Show original image
    subplot(2, length(SNR_levels)+1, 1);
    imshow(I_orig, []); title('Original');
    
    % Test at different SNR levels
    for s = 1:length(SNR_levels)
        SNR = SNR_levels(s);
        
        % Add noise according to SNR
        signalPower = mean(I_orig(:).^2);
        noisePower = signalPower / (10^(SNR/10));
        noise = sqrt(noisePower) * randn(size(I_orig));
        I_noisy = I_orig + noise;
        
        % Show noisy image
        subplot(2, length(SNR_levels)+1, s+1);
        imshow(I_noisy, []);
        title(sprintf('Noisy (SNR=%ddB)', SNR));
        
        % Extract blocks
        Blks = extractBlocks(I_noisy);
        
        % Process each block using GenOMP/OMP
        X = zeros(size(D,2), size(Blks,2));
        for b = 1:size(Blks,2)
            X(:,b) = GenOMP(D, Blks(:,b), T0, epsilon, gOMP_N);
        end
        
        % Reconstruct image
        Rblks = D * X;
        I_denoised = assembleImage(Rblks, [h w]);
        I_denoised = min(max(I_denoised, 0), 1);  % clip to [0,1]
        
        % Show denoised image
        subplot(2, length(SNR_levels)+1, length(SNR_levels)+1+s+1);
        imshow(I_denoised, []);
        title(sprintf('Denoised (SNR=%ddB)', SNR));
        
        % Calculate metrics
        mseDenoising(i, s) = mean((I_denoised(:) - I_orig(:)).^2);
        psnrDenoising(i, s) = 10*log10(1/mseDenoising(i, s));
        
        fprintf('Image %d (%s), SNR=%ddB: MSE=%.6f, PSNR=%.2fdB\n', ...
                i, teFiles(i).name, SNR, mseDenoising(i, s), psnrDenoising(i, s));
    end
    
    % Save figure
    saveas(fig, sprintf('denoising_image%d.png', i));
end

% Plot MSE results for denoising
figure('Name', 'Denoising MSE Results');
bar(mseDenoising);
set(gca, 'XTickLabel', {teFiles.name}, 'XTickLabelRotation', 45);
ylabel('Mean Square Error');
title('Denoising MSE for Different SNR Levels');
legend(arrayfun(@(x) sprintf('%ddB', x), SNR_levels, 'UniformOutput', false));
grid on;
saveas(gcf, 'denoising_mse.png');

%% ------------------ INPAINTING EXPERIMENT ----------------------------
fprintf('\nRunning inpainting experiment...\n');

% Storage for inpainting results
mseInpainting = zeros(numImages, length(SNR_levels));
psnrInpainting = zeros(numImages, length(SNR_levels));

% For each test image
for i = 1:numImages
    % Load and convert to grayscale if needed
    I_orig = im2double(imread(fullfile(testDir,teFiles(i).name)));
    if size(I_orig,3)==3, I_orig = rgb2gray(I_orig); end
    [h, w] = size(I_orig);
    
    % Create a figure for this image
    fig = figure('Name', sprintf('Inpainting - %s', teFiles(i).name));
    
    % Show original image
    subplot(2, length(SNR_levels)+1, 1);
    imshow(I_orig, []); title('Original');
    
    % Define percentage of pixels to remove based on SNR
    % Higher SNR = fewer pixels removed
    missing_percentages = [90, 50, 20, 10]; % Corresponding to SNR levels
    
    % Test at different SNR-equivalent levels
    for s = 1:length(SNR_levels)
        SNR = SNR_levels(s);
        missing_pct = missing_percentages(s);
        
        % Create a binary mask with uniform distribution
        mask = rand(size(I_orig)) > (missing_pct/100);
        I_missing = I_orig .* mask;
        
        % Show image with missing pixels
        subplot(2, length(SNR_levels)+1, s+1);
        imshow(I_missing, []);
        title(sprintf('Missing (%d%%)', missing_pct));
        
        % Extract blocks
        Blks = extractBlocks(I_missing);
        
        % Process each block using GenOMP/OMP
        X = zeros(size(D,2), size(Blks,2));
        for b = 1:size(Blks,2)
            X(:,b) = GenOMP(D, Blks(:,b), T0, epsilon, gOMP_N);
        end
        
        % Reconstruct image
        Rblks = D * X;
        I_inpainted = assembleImage(Rblks, [h w]);
        I_inpainted = min(max(I_inpainted, 0), 1);  % clip to [0,1]
        
        % Show inpainted image
        subplot(2, length(SNR_levels)+1, length(SNR_levels)+1+s+1);
        imshow(I_inpainted, []);
        title(sprintf('Inpainted (%d%%)', missing_pct));
        
        % Calculate metrics
        mseInpainting(i, s) = mean((I_inpainted(:) - I_orig(:)).^2);
        psnrInpainting(i, s) = 10*log10(1/mseInpainting(i, s));
        
        fprintf('Image %d (%s), Missing=%d%%: MSE=%.6f, PSNR=%.2fdB\n', ...
                i, teFiles(i).name, missing_pct, mseInpainting(i, s), psnrInpainting(i, s));
    end
    
    % Save figure
    saveas(fig, sprintf('inpainting_image%d.png', i));
end

% Plot MSE results for inpainting
figure('Name', 'Inpainting MSE Results');
bar(mseInpainting);
set(gca, 'XTickLabel', {teFiles.name}, 'XTickLabelRotation', 45);
ylabel('Mean Square Error');
title('Inpainting MSE for Different Missing Pixel Percentages');
legend(arrayfun(@(x) sprintf('%d%%', x), missing_percentages, 'UniformOutput', false));
grid on;
saveas(gcf, 'inpainting_mse.png');

%% ------------------ SAVE RESULTS -------------------------------------
save('denoising_results.mat', 'mseDenoising', 'psnrDenoising', 'SNR_levels');
save('inpainting_results.mat', 'mseInpainting', 'psnrInpainting', 'missing_percentages');

fprintf('Experiments completed and results saved.\n');
