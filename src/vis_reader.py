"""
Vision Reader
Program for reading sudoku puzzles in through use of edge deterction
and computer vision techniques.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = 'data/img/easy_sudoku_1.jpg'
IMG_SIZE = 360

class puzzle_detector:
    """
    A class to detect and process Sudoku puzzles from images.
    Currently, this class is a placeholder and does not contain any methods or attributes.
    """
    def __init__(self, img):
        self.image = img
        self.puzzle = None
    
    def preprocess_image(self):
        """
        Placeholder method for preprocessing the image.
        This method should be implemented to handle image preprocessing tasks.
        
        Args:
            image (np.ndarray): The input image to preprocess.
        
        Returns:
            np.ndarray: The preprocessed image.
        """
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.resize(self.image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
        self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)

    def detect_corners_canny(self):
        """
        Placeholder method for detecting corners in the Sudoku puzzle using Canny edge detection.
        This method should be implemented to handle corner detection tasks.
        
        Returns:
            list: A list of detected corners.
        """
        # For Canny edge detection:
        # img - Input image. It should be grayscale and float32 type.
        # threshold1 - First threshold for the hysteresis procedure.
        # threshold2 - Second threshold for the hysteresis procedure.
        canny = cv2.Canny(self.image, 120, 255, apertureSize=5)
        
        # Find contours
        corners = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
        # Sort contours by area and keep the largest one
        corners = sorted(corners, key=cv2.contourArea, reverse=True)[:10]
        # Draw contours on the image
        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(canny, corners[0], -1, (0, 255, 0), 3)

        # Convert contours to a more usable format (e.g., list of points)
        corners = [cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True) for c in corners]
        corners = [c.reshape(-1, 2) for c in corners if len(c) >= 4]

        return canny, corners[0] if corners else []
    
    def _sort_points(self, points):
      sorted_points = np.zeros((4, 2), dtype=np.float32)
      sums = points.sum(axis=1)
      diffs = np.diff(points, axis=1).flatten()

      sorted_points[0] = points[np.argmin(sums)]  # Top-left
      sorted_points[2] = points[np.argmax(diffs)]  # Bottom-right
      sorted_points[1] = points[np.argmin(diffs)] # Top-right
      sorted_points[3] = points[np.argmax(sums)] # Bottom-left
      return sorted_points

    def perspective_transform(self, points, show_flag=False):
        """
        Placeholder method for applying a perspective transform to the Sudoku puzzle.
        This method should be implemented to handle perspective transformation tasks.
        
        Returns:
            np.ndarray: The transformed image.
        """
        # Define source and destination points for perspective transform
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        if points.shape[0] != 4 or points.shape[1] != 2:
            raise ValueError("Points must be a 4x2 array.")
        
        # Ensure points are in the correct order (top-left, top-right, bottom-right, bottom-left)
        points = self._sort_points(points)
        
        # Define the destination points for the perspective transform
        pts1 = np.float32(points)
        pts2 = np.float32([[0,0],[IMG_SIZE,0],[0,IMG_SIZE],[IMG_SIZE,IMG_SIZE]])

        # Apply perspective transform
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(self.image,M,(IMG_SIZE,IMG_SIZE))

        
        
        if show_flag:
            # Display the original and transformed images
          # Plot original image with points overlayed
          plt.subplot(121)
          plt.imshow(self.image)
          plt.scatter(pts1[:, 0], pts1[:, 1], color='red', marker='o', label='pts1')
          plt.title('Input')

          # Plot transformed image
          plt.subplot(122)
          plt.imshow(dst)
          plt.title('Transformed')
          plt.show()


        self.image = dst

def read_image(filepath=IMG_PATH):
    """
    Reads an image file containing a Sudoku puzzle.

    Args:
        filepath (str): The path to the image file. Defaults to 'data/images/sudoku_1.png'.

    Returns:
        np.ndarray: The image as a NumPy array.
    """
    image = cv2.imread(filepath)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {filepath}")
    return image

if __name__ == "__main__":
    # Example usage
    try:
        sudoku_image = read_image()
        print("Image read successfully.")
        print("Image shape:", sudoku_image.shape)
    except FileNotFoundError as e:
        print(e)
    
    sudoku = puzzle_detector(sudoku_image)
    sudoku.preprocess_image()

    # Detect corners using Canny edge detection
    img, corners = sudoku.detect_corners_canny()
    
    # Use corners for perspective transform
    sudoku.perspective_transform(corners)

    plt.imshow(sudoku.image, cmap='gray')
    plt.title('Transformed Sudoku Puzzle')
    plt.axis('off')
    plt.show()
    
