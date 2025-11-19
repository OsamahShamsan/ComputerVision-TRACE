# TRACE - Tampered Region Analysis & Copy-Move Exploration

Feature-based Image Forensics: Locate altered regions in copy-move manipulated images.

## Project Overview

Copy-move forgery is a type of image manipulation where a region of an image is copied and pasted to another location in the same image. Feature-based methods detect this by extracting distinctive features and matching similar regions.

## Setup

### Quick Setup (Recommended)

**On macOS/Linux:**
Run the setup script to automatically create/activate the virtual environment and install all dependencies:

```bash
./setup.sh
```

The script will:
- Create virtual environment if it doesn't exist (won't recreate if it exists)
- Activate the virtual environment
- Upgrade pip
- Install all requirements from `requirements.txt`
- Install ipykernel for Jupyter notebook support

**On Windows:**
Use the manual setup steps below.

### Manual Setup

#### Virtual Environment

This project uses a Python virtual environment for dependency management. The virtual environment must be activated before running any project code.

#### Activating the Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

The terminal prompt displays `(venv)` when the virtual environment is active.

#### Deactivating the Virtual Environment

```bash
deactivate
```

#### Installing Dependencies

With the virtual environment activated, dependencies are installed using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains all project dependencies.

**Note:** If you get a `ModuleNotFoundError` for packages like numpy, make sure you've activated the virtual environment and installed requirements.txt.

#### Running Jupyter Notebooks

Jupyter notebooks are used for development and experimentation. With the virtual environment activated:

```bash
# Launch Jupyter Notebook
jupyter notebook

# Or launch JupyterLab (alternative interface)
jupyter lab
```

**Selecting the correct kernel:**
When opening a notebook, make sure to select the kernel `venv (Python X.X.X)` from the kernel menu. This ensures the notebook uses the virtual environment with all installed packages.

If the kernel doesn't appear:
1. Make sure ipykernel is installed: `pip install ipykernel`
2. Register the kernel: `python -m ipykernel install --user --name=venv --display-name "venv (Python $(python --version | cut -d' ' -f2))"`

Notebooks are located in the `notebooks/` directory. The Jupyter interface opens in the default web browser.

#### Creating/Recreating the Virtual Environment

To recreate the virtual environment:

```bash
# Remove old venv (if exists)
rm -rf venv

# Create new venv
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Feature-Based Copy-Move Forgery Detection Methods

### Main Approaches

#### 1. **SIFT (Scale-Invariant Feature Transform)**
   - **Principle**: Extracts keypoints that are invariant to rotation, scale, and illumination
   - **Advantages**: 
     - Robust to geometric transformations
     - Works well with partial occlusions
     - Highly distinctive descriptors
   - **Limitations**:
     - Computationally expensive
     - May miss uniform/smooth regions
   - **Use Case**: Best for images with textured regions

#### 2. **SURF (Speeded-Up Robust Features)**
   - **Principle**: Fast approximation of SIFT using integral images
   - **Advantages**:
     - Faster than SIFT
     - Good scale and rotation invariance
     - More efficient descriptor computation
   - **Limitations**:
     - Less distinctive than SIFT in some cases
     - Still computationally intensive
   - **Use Case**: Balance between speed and accuracy

#### 3. **ORB (Oriented FAST and Rotated BRIEF)**
   - **Principle**: Combines FAST keypoint detector with BRIEF descriptor
   - **Advantages**:
     - Very fast computation
     - Rotation invariant
     - Free and open-source
   - **Limitations**:
     - Less robust to scale changes
     - May produce more false matches
   - **Use Case**: Real-time applications, fast processing needed

#### 4. **AKAZE (Accelerated-KAZE)**
   - **Principle**: Uses nonlinear scale spaces with efficient computation
   - **Advantages**:
     - Better localization than SIFT
     - Robust to noise
     - Good performance on textured images
   - **Limitations**:
     - Moderate computational cost
     - Less widely used (fewer implementations)
   - **Use Case**: High-quality detection in textured images

#### 5. **BRISK (Binary Robust Invariant Scalable Keypoints)**
   - **Principle**: Binary descriptor with scale-space keypoint detection
   - **Advantages**:
     - Fast matching (Hamming distance)
     - Scale and rotation invariant
     - Good for real-time applications
   - **Limitations**:
     - Less distinctive for smooth regions
     - Binary descriptor limitations
   - **Use Case**: Fast detection in real-time systems

#### 6. **FREAK (Fast Retina Keypoint)**
   - **Principle**: Biologically inspired binary descriptor
   - **Advantages**:
     - Very fast matching
     - Good rotation invariance
     - Efficient for large datasets
   - **Limitations**:
     - Requires good keypoint detection
     - Less robust than floating-point descriptors
   - **Use Case**: Speed-critical applications

#### 7. **Dense Feature Extraction Methods**
   - **Principle**: Extract features from all image regions (not just keypoints)
   - **Methods**:
     - Dense SIFT (DSIFT)
     - Dense SURF
     - Patch-based descriptors
   - **Advantages**:
     - Can detect regions with few keypoints
     - More comprehensive coverage
   - **Limitations**:
     - Higher computational cost
     - More false positives possible
   - **Use Case**: Images with smooth or uniform regions

### Post-Processing Steps

#### 1. **Feature Matching**
   - **Brute Force Matching**: Compare all descriptors (slow but accurate)
   - **FLANN (Fast Library for Approximate Nearest Neighbors)**: Fast approximate matching
   - **Ratio Test (Lowe's)**: Filter matches using nearest neighbor ratio
   - **Geometric Filtering**: Remove matches using RANSAC or homography constraints

#### 2. **Clustering and Grouping**
   - **DBSCAN**: Cluster matched features to identify copied regions
   - **Spatial Grouping**: Group features based on spatial proximity
   - **Hough Transform**: Detect geometric patterns in matches

#### 3. **Region Extraction**
   - **Bounding Boxes**: Draw rectangles around matched clusters
   - **Convex Hull**: Find boundaries of tampered regions
   - **Segmentation**: Use watershed or graph-cut for precise boundaries

#### 4. **Refinement**
   - **Morphological Operations**: Clean up detection masks
   - **Region Growing**: Expand detected regions
   - **False Positive Reduction**: Apply size/ratio filters

### Hybrid Approaches

#### 1. **Multi-Scale Feature Detection**
   - Extract features at multiple scales
   - Combine results for better coverage

#### 2. **Feature Fusion**
   - Combine multiple feature types (e.g., SIFT + SURF)
   - Use ensemble methods for better accuracy

#### 3. **Learning-Based Features**
   - **Deep Learning**: CNN-based features
   - **Learned Descriptors**: Train feature extractors
   - **Hybrid Traditional + Deep**: Combine handcrafted and learned features

## Performance Metrics

- **True Positive Rate (TPR)**: Correctly detected tampered regions
- **False Positive Rate (FPR)**: Incorrectly flagged regions
- **F1-Score**: Harmonic mean of precision and recall
- **Computational Time**: Speed of detection
- **Localization Accuracy**: Precision of boundary detection

## Implementation Considerations

### 1. **Preprocessing**
   - Image normalization
   - Noise reduction
   - Color space conversion (RGB, HSV, LAB)

### 2. **Feature Extraction Parameters**
   - Keypoint detection thresholds
   - Descriptor size
   - Scale space parameters

### 3. **Matching Strategies**
   - Distance metrics (Euclidean, Hamming, Cosine)
   - Match filtering criteria
   - Geometric consistency checks

### 4. **Optimization**
   - Parallel processing
   - GPU acceleration
   - Efficient data structures (KD-trees, hash tables)

## Recommended Pipeline

The standard detection pipeline consists of:

1. **Preprocessing**: Image normalization and optional grayscale conversion
2. **Feature Extraction**: Method selection based on image characteristics
3. **Feature Matching**: Ratio test and geometric filtering applied
4. **Clustering**: Spatial grouping of matched features
5. **Region Detection**: Boundary extraction of tampered regions
6. **Post-processing**: Result refinement and visualization

## Tool Recommendations

- **OpenCV**: Comprehensive feature detection implementations
- **scikit-image**: Additional image processing tools
- **NumPy/SciPy**: Numerical computations
- **Matplotlib**: Visualization
- **scikit-learn**: Clustering algorithms (DBSCAN)

## Machine Learning Approaches

### Supervised Learning Methods

Supervised learning approaches for copy-move forgery detection require labeled datasets with ground truth annotations.

#### CNN-Based Detection
- **Patch-Based CNNs**: Training convolutional neural networks on image patches to classify tampered and authentic regions
- **Fully Convolutional Networks (FCN)**: Pixel-level segmentation networks for precise region localization
- **Encoder-Decoder Architectures**: U-Net and similar architectures for boundary detection
- **Siamese Networks**: Learning similarity metrics between image regions to identify copied areas

#### Transfer Learning
- **Pre-trained Feature Extractors**: Using ImageNet pre-trained CNNs (VGG, ResNet, EfficientNet) as feature extractors
- **Fine-tuning**: Adapting pre-trained models on forensic-specific datasets
- **Multi-task Learning**: Joint training for detection and localization

#### Hybrid Supervised Methods
- **Traditional + CNN Features**: Combining handcrafted features (SIFT, SURF) with learned CNN features
- **Ensemble Methods**: Stacking multiple CNN models with traditional feature-based methods

### Unsupervised Learning Methods

Unsupervised approaches do not require labeled training data, making them suitable for scenarios with limited ground truth.

#### Autoencoders
- **Variational Autoencoders (VAE)**: Learning normal image representations to identify anomalies
- **Convolutional Autoencoders**: Reconstructing images and detecting inconsistencies in reconstruction errors
- **Adversarial Autoencoders**: Using adversarial training to learn robust feature representations

#### Self-Supervised Learning
- **Contrastive Learning**: Learning representations by contrasting similar and dissimilar image patches
- **Rotation/Transformation Prediction**: Pretext tasks for feature learning without labels
- **Masked Image Modeling**: Similar to vision transformers, predicting masked regions

#### Clustering-Based Methods
- **Feature Clustering**: Using unsupervised clustering (K-means, DBSCAN) on learned features
- **Spectral Clustering**: Grouping similar regions in feature space
- **Graph Neural Networks**: Unsupervised graph-based methods for region grouping

### Semi-Supervised Methods

Leveraging both labeled and unlabeled data for improved performance.

- **Pseudo-labeling**: Generating labels for unlabeled data using model predictions
- **Consistency Regularization**: Enforcing consistency across different augmentations
- **Co-training**: Training multiple models on different feature views

### Deep Learning Architectures

#### Detection Networks
- **YOLO-style Networks**: Single-stage detectors for real-time copy-move detection
- **Faster R-CNN**: Two-stage detection with region proposal networks
- **Mask R-CNN**: Instance segmentation for precise region boundaries

#### Attention Mechanisms
- **Self-Attention**: Identifying self-similarities within images
- **Spatial Attention**: Focusing on suspicious regions
- **Transformer-Based Models**: Vision Transformers (ViT) adapted for copy-move detection

### Advantages of ML/CNN Approaches

- **Automatic Feature Learning**: No manual feature engineering required
- **Robustness**: Better handling of post-processing attacks (compression, blur, rotation)
- **End-to-End Learning**: Direct mapping from images to detection masks
- **Generalization**: Potential for handling various tampering scenarios

### Limitations of ML/CNN Approaches

- **Data Requirements**: Large labeled datasets needed for supervised methods
- **Computational Cost**: Training and inference can be computationally expensive
- **Interpretability**: Less interpretable than traditional feature-based methods
- **Overfitting Risk**: May not generalize well to unseen manipulation types

### Hybrid Approaches

Combining traditional feature-based methods with machine learning:

- **Feature Extraction + Classifier**: Using traditional features (SIFT, SURF) with ML classifiers (SVM, Random Forest, Neural Networks)
- **CNN Feature Extraction**: Replacing handcrafted features with CNN-extracted features in traditional pipelines
- **Ensemble Methods**: Combining predictions from both traditional and deep learning models

## Research Directions

- **Deep Learning**: End-to-end copy-move detection networks
- **Attention Mechanisms**: Focus on suspicious regions
- **Transformers**: Self-similarity detection in images
- **Multi-modal Features**: Combine traditional and learned features
- **Robustness**: Handling post-processing (blur, compression, rotation)
- **Few-Shot Learning**: Adapting models to new manipulation types with minimal data
- **Domain Adaptation**: Transferring knowledge across different image domains

## Usage

### Running Experiments

All experiments and code development are conducted in Jupyter notebooks located in the `notebooks/` directory.

1. Activate the virtual environment
2. Launch Jupyter: `jupyter notebook` or `jupyter lab`
3. Open the desired notebook from the `notebooks/` directory
4. Execute cells to run experiments and view results

### Configuration

Project parameters are managed through configuration files in the `configs/` directory. Modify these files to adjust experiment parameters without editing code directly.

### Output Organization

- Processed data: `data/processed/`
- Experiment results: `data/results/`
- Generated visualizations: `outputs/`
- Saved models: `models/`

### Dataset Setup

Place your raw dataset in the `data/raw/` directory. The data setup notebook will:

1. Inspect dataset structure and statistics
2. Find image-ground truth pairs
3. Split dataset into train/val/test sets
4. Organize processed data in workspace

**To use the data setup notebook**:
- Open `notebooks/01_data_inspection_and_setup.ipynb`
- Ensure raw data is in `data/raw/` directory
- Run cells to process and organize dataset

### Dataset Category Selection

The FAU dataset contains many categories (30GB total). See `DATASET_GUIDE.md` for recommendations on which categories to download:

- **Essential**: `nul`, `rot`, `scale` (~5-8GB)
- **Recommended**: Add `cmb_easy1`, `lnoise`, `jpeg` (~10-15GB)
- **Skip**: `orig`, `*_sd` patterns, `scale_down` (not needed for training)

Configure category filtering in `configs/data_config.json` before downloading.

### Dataset Sufficiency and Alternatives

The FAU dataset contains 48 base images with various alterations. While sufficient for feature-based methods, deep learning approaches may benefit from additional datasets:

- **FAU Dataset**: Good for feature-based methods, evaluation, and transfer learning
- **For more diversity**: See `DATASET_RECOMMENDATIONS.md` for alternative datasets:
  - **CoMoFoD**: 200 forged + 200 authentic images
  - **CASIA v2.0**: 12,614 images (largest available)
  - **MICC-F220/F600**: 220-600 images with ready train/test splits

**Recommendation**: Start with FAU dataset, then combine with CoMoFoD or MICC datasets for deep learning training if needed.


## Contributing

Contributions follow standard development practices:

1. Virtual environment activation before making changes
2. Dependency management via `pip install -r requirements.txt`
3. Testing of changes before committing
4. Adherence to project coding standards

## License


