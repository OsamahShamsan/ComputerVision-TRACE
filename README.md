# TRACE - Tampered Region Analysis & Copy-Move Exploration

Feature-based Image Forensics: Locate altered regions in copy-move manipulated images.

## Project Overview

Copy-move forgery is a type of image manipulation where a region of an image is copied and pasted to another location in the same image. Feature-based methods detect this by extracting distinctive features and matching similar regions.

## Setup

### Virtual Environment

This project uses a Python virtual environment for dependency management. **Always activate the venv before working on this project.**

#### Activating the Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

When activated, you should see `(venv)` in your terminal prompt.

#### Deactivating the Virtual Environment

```bash
deactivate
```

#### Installing Dependencies

After activating the venv, install required packages:

```bash
pip install -r requirements.txt
```

(If `requirements.txt` doesn't exist yet, create it with your project dependencies)

#### Creating/Recreating the Virtual Environment

If you need to recreate the virtual environment:

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

1. **Preprocessing**: Normalize image, convert to grayscale (if needed)
2. **Feature Extraction**: Choose method based on image characteristics
3. **Feature Matching**: Use ratio test and geometric filtering
4. **Clustering**: Group matched features spatially
5. **Region Detection**: Extract boundaries of tampered regions
6. **Post-processing**: Refine and visualize results

## Tool Recommendations

- **OpenCV**: Comprehensive feature detection implementations
- **scikit-image**: Additional image processing tools
- **NumPy/SciPy**: Numerical computations
- **Matplotlib**: Visualization
- **scikit-learn**: Clustering algorithms (DBSCAN)

## Research Directions

- **Deep Learning**: End-to-end copy-move detection networks
- **Attention Mechanisms**: Focus on suspicious regions
- **Transformers**: Self-similarity detection in images
- **Multi-modal Features**: Combine traditional and learned features
- **Robustness**: Handling post-processing (blur, compression, rotation)

## Usage

*Note: Usage instructions will be added as the implementation progresses.*

## Contributing

1. Always activate the virtual environment before making changes
2. Install/update dependencies using `pip install -r requirements.txt`
3. Test your changes before committing
4. Follow the project's coding standards

## License

*Add license information here*

