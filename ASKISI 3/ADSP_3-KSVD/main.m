%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  MAIN SCRIPT  (save as main.m)                                       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Trains on *all* images in dataset/train, reconstructs every image in
%  dataset/test, reports PSNR/MSE/sparsity, visualises the dictionary,
%  and saves D, Xtrain, Xtest, Ytrain plus stats and PNG snapshots.
%-------------------------------------------------------------------------
%  IMPORTANT ➜ **Do not ℓ2‑normalise the signal blocks**; only the
%  dictionary atoms are unit‑norm.  This preserves absolute brightness
%  and fixes low‑contrast artefacts.
%-------------------------------------------------------------------------
%  Author:  Angelos‑Nikolaos Potamianos · AM 1084537 · May 2025
%=========================================================================

clear; close all; clc;

%% ------------------ PARAMETERS ----------------------------------------
blockSize = 8;        % patch dimension (pixels)
K         = 512;      % dictionary atoms (K ≥ blockSize^2)
T0        = 5;        % sparsity per block (try 5‑10)
epsilon   = 0;        % residual threshold (0 ➜ off)
noEpochs  = 100;       % dictionary learning epochs
gOMP_N    = 1;        % =1 → OMP, >1 → GenOMP batch size
maxViz    = 7;        % how many test images to show/save
rng('default');       % reproducibility seed

%% ------------------ DATASET PATHS ------------------------------------
trainDir = fullfile('dataset','train');
testDir  = fullfile('dataset','test');
imgExt   = {'*.png','*.jpg','*.bmp'};

trFiles = struct([]); teFiles = struct([]);
for e = 1:numel(imgExt)
    trFiles = [trFiles; dir(fullfile(trainDir,imgExt{e}))]; %#ok<AGROW>
    teFiles = [teFiles; dir(fullfile(testDir ,imgExt{e}))]; %#ok<AGROW>
end
assert(~isempty(trFiles),'No training images found.');
assert(~isempty(teFiles) ,'No test images found.');
fprintf('Found %d training and %d test images.\n',numel(trFiles),numel(teFiles));

%% ------------------ HELPERS ------------------------------------------
extractBlocks = @(I) im2col(I,[blockSize blockSize],'distinct');
assembleImage = @(B,imSize) col2im(B,[blockSize blockSize],imSize,'distinct');

%% ------------------ BUILD Ytrain -------------------------------------
fprintf('Extracting %dx%d blocks from training set...\n',blockSize,blockSize);
Ytrain = [];
for i = 1:numel(trFiles)
    I = im2double(imread(fullfile(trainDir,trFiles(i).name)));
    if size(I,3)==3, I = rgb2gray(I); end
    Ytrain = [Ytrain, extractBlocks(I)]; %#ok<AGROW>
end

%% ------------------ INITIAL DICTIONARY -------------------------------
D0 = normalizeDictionary(randn(blockSize^2,K));  % only atoms are ℓ2‑norm

%% ------------------ DICTIONARY LEARNING ------------------------------
[D,Xtrain,mseHist] = DictionaryLearning(D0,Ytrain,T0,epsilon,noEpochs,gOMP_N);

figure; plot(1:noEpochs,mseHist,'-o','LineWidth',1.5); grid on;
xlabel('Epoch'); ylabel('MSE'); title('Training reconstruction error');

%% ------------------ VISUALISE DICTIONARY -----------------------------
dispD(D,blockSize);

%% ------------------ RECONSTRUCT TEST IMAGES --------------------------
psnrVals  = zeros(numel(teFiles),1);
mseVals   = zeros(numel(teFiles),1);
meanSpars = zeros(numel(teFiles),1);
XtestCell = cell(numel(teFiles),1);
IrecViz   = cell(maxViz,1);  % store first few reconstructions for display

fprintf('Reconstructing test images...\n');
parfor i = 1:numel(teFiles)
    % ---- load & greyscale -------------------------------------------
    I = im2double(imread(fullfile(testDir,teFiles(i).name)));
    if size(I,3)==3, I = rgb2gray(I); end
    [h,w] = size(I);

    % ---- sparse code every block ------------------------------------
    Blks = extractBlocks(I);             % <64 × #blocks>
    Xtmp = zeros(K,size(Blks,2));
    for b = 1:size(Blks,2)
        Xtmp(:,b) = GenOMP(D,Blks(:,b),T0,epsilon,gOMP_N);
    end
    Rblks = D * Xtmp;                    % reconstructed blocks
    Irec  = assembleImage(Rblks,[h w]);  % back to image
    Irec  = min(max(Irec,0),1);          % clip [0,1]

    % ---- metrics -----------------------------------------------------
    mseVals(i)  = mean((Irec(:)-I(:)).^2);
    psnrVals(i) = 10*log10(1/mseVals(i));
    meanSpars(i)= mean(sum(Xtmp~=0,1));
    XtestCell{i}= Xtmp; %#ok<NASGU>

    % ---- store viz frames locally (parfor‑safe) ----------------------
    if i<=maxViz
        IrecViz{i} = Irec;          %#ok<PARFOR>
    end
end

%% ------------- DISPLAY / SAVE FIRST FEW FIGURES (outside PARFOR) ----
for v = 1:min(maxViz,numel(teFiles))
    Iori = im2double(imread(fullfile(testDir,teFiles(v).name)));
    if size(Iori,3)==3, Iori = rgb2gray(Iori); end
    f = figure('Name',sprintf('Recon %s',teFiles(v).name));
    subplot(1,2,1); imshow(Iori,[]); title(sprintf('Original – %s',teFiles(v).name),'Interpreter','none');
    subplot(1,2,2); imshow(IrecViz{v},[]); title(sprintf('Reconstruction – PSNR %.2f dB',psnrVals(v)));
    saveas(f,sprintf('test_recon_%02d.png',v));
end

%% ------------------ SUMMARY ------------------------------------------
fprintf('\nTest‑set summary over %d images:\n',numel(teFiles));
fprintf('  Average PSNR : %.2f dB (std %.2f)\n',mean(psnrVals),std(psnrVals));
fprintf('  Average MSE  : %.4e\n',mean(mseVals));
fprintf('  Mean sparsity: %.2f coefficients per block\n',mean(meanSpars));

%% ------------------ SAVE OUTPUTS -------------------------------------
save('D.mat','D');
save('Xtrain.mat','Xtrain');
save('Ytrain.mat','Ytrain');
save('Xtest.mat' ,'XtestCell','-v7.3');
save('Ytest_stats.mat','psnrVals','mseVals','meanSpars');

fprintf('Saved D.mat, Xtrain.mat, Xtest.mat, Ytrain.mat, and stats.\n');
