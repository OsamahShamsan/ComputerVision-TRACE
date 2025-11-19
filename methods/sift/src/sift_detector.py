"""
SIFT-based copy-move forgery detection.

Extracts SIFT features, matches similar features, and generates binary masks
showing tampered regions.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class SIFTDetector:
    """
    SIFT-based copy-move forgery detector.
    
    Extracts SIFT features from image, matches similar features,
    and generates binary mask showing tampered regions.
    """
    
    def __init__(self, n_features: int = 0, contrast_threshold: float = 0.04,
                 edge_threshold: int = 10, sigma: float = 1.6):
        """
        Initialize SIFT detector.
        
        Parameters:
            n_features: Maximum number of features to extract (0 = unlimited)
            contrast_threshold: Threshold for feature detection (lower = more features)
            edge_threshold: Threshold for edge detection (higher = fewer edge features)
            sigma: Gaussian blur sigma for first octave
        """
        self.n_features = n_features
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        
        # Create SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
        
        # FLANN matcher for feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract SIFT features from image.
        
        Parameters:
            image: Input image (BGR or grayscale)
            
        Returns:
            keypoints: Detected keypoints
            descriptors: Feature descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Extract SIFT features
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            logger.warning("No SIFT features detected")
            return [], np.array([])
        
        logger.info(f"Extracted {len(keypoints)} SIFT features")
        return keypoints, descriptors
    
    def match_features(self, descriptors1: np.ndarray, descriptors2: np.ndarray,
                      ratio_threshold: float = 0.75) -> List[cv2.DMatch]:
        """
        Match features between two descriptor sets.
        Uses ratio test to filter good matches.
        
        Parameters:
            descriptors1: First set of descriptors
            descriptors2: Second set of descriptors (same image for copy-move)
            ratio_threshold: Ratio test threshold (Lowe's ratio test)
            
        Returns:
            good_matches: List of good feature matches
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if len(descriptors1) < 2 or len(descriptors2) < 2:
            return []
        
        # Match features using KNN (k=2 for ratio test)
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test (Lowe's ratio test)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        logger.info(f"Found {len(good_matches)} good matches (ratio test: {ratio_threshold})")
        return good_matches
    
    def detect_copy_move(self, image: np.ndarray, 
                        ratio_threshold: float = 0.75,
                        min_match_count: int = 4,
                        distance_threshold: float = 50.0) -> Tuple[np.ndarray, dict]:
        """
        Detect copy-move forgery in image using SIFT features.
        
        Parameters:
            image: Input image (BGR format)
            ratio_threshold: Ratio test threshold for feature matching
            min_match_count: Minimum number of matches to consider region tampered
            distance_threshold: Maximum distance between matched features (pixels)
            
        Returns:
            binary_mask: Binary mask (255 = tampered, 0 = authentic)
            stats: Dictionary with detection statistics
        """
        # Extract features
        keypoints, descriptors = self.extract_features(image)
        
        if descriptors is None or len(descriptors) < 2:
            logger.warning("Not enough features for copy-move detection")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            return mask, {'matches': 0, 'keypoints': 0}
        
        # Match features (same image, so descriptors1 = descriptors2)
        matches = self.match_features(descriptors, descriptors, ratio_threshold)
        
        if len(matches) < min_match_count:
            logger.warning(f"Not enough matches ({len(matches)}) for detection")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            return mask, {'matches': len(matches), 'keypoints': len(keypoints)}
        
        # Filter matches by distance (remove matches that are too close - likely same region)
        filtered_matches = []
        for match in matches:
            pt1 = keypoints[match.queryIdx].pt
            pt2 = keypoints[match.trainIdx].pt
            distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            if distance > distance_threshold:  # Matches far apart indicate copy-move
                filtered_matches.append(match)
        
        logger.info(f"Filtered to {len(filtered_matches)} matches (distance > {distance_threshold})")
        
        # Create binary mask from matched regions
        mask = self._matches_to_mask(image, keypoints, filtered_matches)
        
        stats = {
            'keypoints': len(keypoints),
            'matches': len(matches),
            'filtered_matches': len(filtered_matches),
            'ratio_threshold': ratio_threshold,
            'distance_threshold': distance_threshold
        }
        
        return mask, stats
    
    def _matches_to_mask(self, image: np.ndarray, keypoints: List,
                        matches: List[cv2.DMatch]) -> np.ndarray:
        """
        Convert feature matches to binary mask.
        
        Parameters:
            image: Input image
            keypoints: Detected keypoints
            matches: Feature matches
            
        Returns:
            binary_mask: Binary mask showing tampered regions
        """
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        if len(matches) == 0:
            return mask
        
        # Get matched point pairs
        matched_points = []
        for match in matches:
            pt1 = keypoints[match.queryIdx].pt
            pt2 = keypoints[match.trainIdx].pt
            matched_points.append((pt1, pt2))
        
        # Draw regions around matched points
        for pt1, pt2 in matched_points:
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            
            # Draw circles around matched points
            radius = 20  # Radius for marking tampered regions
            cv2.circle(mask, (x1, y1), radius, 255, -1)
            cv2.circle(mask, (x2, y2), radius, 255, -1)
            
            # Draw line connecting matched points
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
        
        return mask


def load_image(image_path: Path) -> np.ndarray:
    """
    Load image from file path.
    
    Parameters:
        image_path: Path to image file
        
    Returns:
        image: Image as numpy array (BGR format)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    return image


def save_mask(mask: np.ndarray, output_path: Path) -> None:
    """
    Save binary mask to file.
    
    Parameters:
        mask: Binary mask (0 or 255)
        output_path: Path to save mask
    """
    cv2.imwrite(str(output_path), mask)

