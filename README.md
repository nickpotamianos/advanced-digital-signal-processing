# Digital Signal Processing Applications (ADSP)
*Εφαρμογές Ψηφιακής Επεξεργασίας Σημάτων*

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020a+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

This repository contains implementations and assignments for the Digital Signal Processing Applications course at the University of Patras, Department of Computer Engineering and Informatics (CEID).

## 📚 Course Overview

The course covers advanced topics in digital signal processing with practical implementations focusing on:
- Spectral content estimation and frequency analysis
- Low-rank modeling and matrix factorization techniques
- Dictionary learning and sparse coding (K-SVD)
- Linear Discriminant Analysis (LDA)
- Principal Component Analysis (PCA) and Eigenfaces

## 🗂️ Repository Structure

### Assignment 1: Spectral Content Estimation (`ASKISI 1/`)
Implementation of spectral estimation techniques including:
- **MUSIC Algorithm**: Multiple Signal Classification for frequency estimation
- **Eigenvalue Methods**: Statistical analysis of signal subspaces
- **Parameter Studies**: Analysis of different array sizes (M=2,10,20,30,40,50) and noise levels

**Key Files:**
- `Askisi1_5.m` - Basic spectral estimation
- `Askisi1_6.m` - MUSIC algorithm implementation
- `1_7/Askisi1_7.m` - Parameter analysis with visualization
- `1_8/Askisi1_8.m` - Noise sensitivity analysis
- Generated plots showing eigenvalue distributions, MUSIC spectra, and Q-functions

### Assignment 2: Low Rank Modeling & Eigenfilters (`ASKISI 2/`)
Exploration of low-rank approximation techniques for signal processing applications.

### Assignment 3: K-SVD Dictionary Learning (`ASKISI 3/`)
Implementation of the K-SVD algorithm for sparse representation and dictionary learning.

### Assignment 4: Linear Discriminant Analysis (`ASKISI 4/`)
Application of LDA techniques for dimensionality reduction and classification.

### Assignment 5: PCA & Eigenfaces (`ASKISI 5/`)
Comprehensive implementation of Principal Component Analysis with applications to:

#### Face Recognition System
- **Dataset**: 40 subjects × 10 images per subject (400 total images)
- **Method**: PCA-based Eigenfaces approach
- **Features**: Automatic train/test split, performance evaluation, visualization

**Core Components:**
```
eigenfaces.py          # Complete Python implementation
eigenfaces.ipynb       # Jupyter notebook version
faces_dataset/         # Face images organized by subject (s1-s40)
split.json            # Train/test split configuration
```

#### Video Processing & Low-Rank Reconstruction
- **Datasets**: Bootstrap, Campus, Lobby videos (.mat format)
- **Techniques**: SVD-based reconstruction, batch processing, PCA with centering

**MATLAB Utilities:**
```
frames2matrix.m        # Convert video frames to matrix format
matrix2frames.m        # Convert matrix back to video sequence
lowrank_reconstruct.m  # SVD-based low-rank approximation
svd_pca_videos.m      # Main video processing script
```

#### Video Analysis Tools
Advanced comparison and visualization tools for reconstruction quality assessment:

```
Video Results/
├── videocompare.py     # Python video comparison script
└── videocompare.ipynb  # Interactive notebook for analysis
```

**Features:**
- Real-time side-by-side video comparison
- Quantitative metrics (MSE, PSNR, SSIM)
- Difference heatmap visualization
- Statistical analysis and plotting

## 🚀 Getting Started

### Prerequisites

**MATLAB Requirements:**
- MATLAB R2020a or newer
- Signal Processing Toolbox
- Image Processing Toolbox

**Python Requirements:**
```bash
pip install numpy scipy matplotlib scikit-learn opencv-python pandas jupyter
pip install scikit-image  # For SSIM calculations
```

### Quick Start Guide

#### 1. Face Recognition (Assignment 5)
```bash
# Navigate to ASKISI 5 directory
cd "ASKISI 5"

# Run the complete eigenfaces implementation
python eigenfaces.py

# Or use the interactive notebook
jupyter notebook eigenfaces.ipynb
```

**Configuration Options:**
- Modify `K_EIGENFACES` to change the number of principal components
- Set `MAKE_CURVE = True` to generate performance vs. k plots
- Adjust `SHOW_VISUALIZATIONS` for educational plots

#### 2. Video Processing
```matlab
% In MATLAB, navigate to ASKISI 5
cd('ASKISI 5')

% Run video reconstruction analysis
svd_pca_videos
```

#### 3. Spectral Analysis (Assignment 1)
```matlab
% Navigate to ASKISI 1
cd('ASKISI 1')

% Run MUSIC algorithm analysis
Askisi1_6

% Run parameter sensitivity analysis
cd('1_7')
Askisi1_7
```

## 📊 Key Results & Visualizations

### Eigenfaces Analysis
- **Classification Reports**: Precision, recall, F1-score per subject
- **Performance Curves**: Accuracy and macro-F1 vs. number of components
- **Eigenface Visualization**: Principal components as face-like patterns
- **Confusion Matrices**: Detailed classification performance

### Video Reconstruction
- **Quality Metrics**: MSE, PSNR, SSIM comparisons
- **Reconstruction Videos**: Original vs. low-rank approximations
- **Method Comparison**: SVD vs. PCA vs. batch processing
- **Real-time Analysis**: Interactive video comparison tool

### Spectral Estimation
- **MUSIC Spectra**: High-resolution frequency estimation
- **Eigenvalue Analysis**: Statistical properties of correlation matrices
- **Parameter Studies**: Effect of array size and noise on performance
- **Histogram Analysis**: Distribution of eigenvalues and frequencies

## 🛠️ Technical Details

### Face Recognition Implementation
The eigenfaces system implements:
- Automatic mean centering and normalization
- Efficient covariance matrix computation
- Configurable train/test splits (default: images 1-8 train, 9-10 test)
- Multiple distance metrics for classification
- Comprehensive performance evaluation

### Video Processing Pipeline
1. **Data Loading**: Flexible support for .mat files with various structures
2. **Preprocessing**: RGB to grayscale conversion, normalization
3. **Reconstruction**: SVD/PCA-based low-rank approximation
4. **Evaluation**: Multiple quality metrics and visualization tools

### MATLAB Integration
- Cross-platform compatibility (Windows/macOS/Linux)
- Efficient matrix operations using MATLAB's optimized libraries
- Modular design for easy parameter modification
- Comprehensive plotting and visualization

## 📈 Performance Benchmarks

### Face Recognition Results
- **Typical Accuracy**: 85-95% with k=32 eigenfaces
- **Optimal Components**: Usually 20-50 for 40-subject dataset
- **Training Time**: < 5 seconds for full dataset
- **Classification Time**: < 1ms per face

### Video Reconstruction Quality
- **Rank-10 SVD**: PSNR typically 25-35 dB
- **Rank-3 SVD**: PSNR typically 20-30 dB
- **PCA vs SVD**: Comparable quality with centering
- **Batch Processing**: Maintains quality with lower memory usage

## 🤝 Contributing

This repository serves educational purposes for the ADSP course. Contributions should maintain:
- Clear documentation and comments (Greek or English)
- Consistent coding style
- Proper attribution for external libraries
- Educational value and clarity

## 📄 License

This project is created for academic purposes at the University of Patras. Please respect academic integrity guidelines when using this code.

## 🙋‍♂️ Support

For questions about the implementations or course material:
- Check the course documentation in each assignment folder
- Review the extensive comments in the code files
- Consult the generated reports and visualizations

## 🔗 References

- [MUSIC Algorithm](https://en.wikipedia.org/wiki/MUSIC_(algorithm))
- [Eigenfaces for Recognition](http://www.face-rec.org/algorithms/PCA/jcn.pdf)
- [K-SVD Dictionary Learning](https://elad.cs.technion.ac.il/software/)
- [Low-Rank Matrix Approximation](https://web.stanford.edu/group/mmds/slides2010/Lecture3.pdf)

---

**Course**: Digital Signal Processing Applications  
**Institution**: University of Patras, Department of Computer Engineering & Informatics  
**Academic Year**: 2024-2025
