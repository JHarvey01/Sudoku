"""
Vision Reader
Program for reading sudoku puzzles in through use of edge deterction
and computer vision techniques.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import torch
from torchvision.models import resnet18
from torchvision.transforms import v2
from classes import number_square, sudoku_puzzle

IMG_PATH = 'data/img/sudoku/easy_sudoku_1.jpg'
IMG_SIZE = 360

MODEL_PATH = 'data/models/resnet18/number_detection_model_2025-06-27_16-47.pth'

CELL_POPULATED_THRESH = 20  # Threshold to determine if a cell is valid (not empty) number of pixels

class puzzle_detector:
	"""
	A class to detect and process Sudoku puzzles from images.
	Currently, this class is a placeholder and does not contain any methods or attributes.
	"""
	def __init__(self, img):
		self.image = img
	
	def preprocess_image(self):
		"""
		Preprocess the input image for Sudoku puzzle detection. Modifies the image in place.
		"""
		self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		self.image = cv2.resize(self.image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

		self.image = cv2.GaussianBlur(self.image, (3, 3), 0)
		self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4)

	def detect_corners_canny(self, show_flag=0):
		"""
		Detect corners in the Sudoku puzzle using Canny edge detection.
		
		Args:
			show_flag (int): Flag to indicate whether to display the detected corners. Defaults to 0 (do not display).

		Returns:
			list: A list of detected corners.
		"""
		# For Canny edge detection:
		# img - Input image. It should be grayscale and float32 type.
		# threshold1 - First threshold for the hysteresis procedure.
		# threshold2 - Second threshold for the hysteresis procedure.
		
		img = self.image.copy()
		canny = cv2.Canny(img, 120, 255, apertureSize=7)
		# Dilate the edges to make them more pronounced
		canny = cv2.dilate(canny, None)
		# Find contours
		corners = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
		# Sort contours by area and keep the largest one
		corners = sorted(corners, key=cv2.contourArea, reverse=True)[:10]
		# Draw contours on the image
		canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
		for c in corners:
			cv2.drawContours(canny, c, -1, (0, 255, 0), 3)
		# cv2.drawContours(canny, corners[0], -1, (0, 255, 0), 3)

		if show_flag:
			plt.imshow(canny, cmap='gray')
			plt.title('Canny Edges')
			plt.show()

		# Convert contours to a more usable format (e.g., list of points)
		corners = [cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True) for c in corners]
		corners = [c.reshape(-1, 2) for c in corners if len(c) >= 4]

		return canny, corners[0] if corners else []
	
	def _sort_points(self, points):
		"""		
		Sorts the corner points in a specific order: top-left, top-right, bottom-right, bottom-left.
		
		Args:
			points (np.ndarray): An array of points to be sorted, expected to be of shape (4, 2).
		"""
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
		Applies a perspective transform to the Sudoku puzzle image based on the provided corner points.

		Args:
			points (list or np.ndarray): A list or array of four corner points in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
			show_flag (bool): Flag to indicate whether to display the original and transformed images. Defaults to False.
		
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

class number_identifier:
	"""
	A class to identify numbers in Sudoku cells using a pre-trained model.
	"""
	
	# Preprocessing pipeline for the input images
	preprocess = v2.Compose([
		v2.ToImage(),  # Converts ndarray to tensor
		v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	
	def __init__(self, img):
		self.image = img
		self.cells = []
		self.split_image()
		self.model = None

	def load_model(self):
		"""
		Loads a pre-trained model for digit classification.
		
		Returns:
			int: 0 if the model is loaded successfully, 1 otherwise.
		"""
		# This is a placeholder implementation
		# Actual implementation would involve loading a trained model
		print(f"Loading model: {MODEL_PATH}...")
		model = resnet18(weights=None)  # Assuming a ResNet18 model for digit classification
		model.fc = torch.nn.Linear(model.fc.in_features, 10)
		
		state_dict = torch.load(MODEL_PATH)
		model.load_state_dict(state_dict)
		
		model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
		if model:
			print("Model loaded successfully.")
			model.eval()
			self.model = model
			return 0
		else:
			print("Failed to load model.")
		return 1
	

	def get_cell_val(self, cell):
		"""
		Extracts the value from a Sudoku cell.

		Args:
			cell (np.ndarray): The image of the Sudoku cell.
		
		Returns:
			int: The value extracted from the cell.
		"""
		# This is a placeholder implementation
		# Actual implementation would involve OCR or similar techniques
		if self.model == None:
			print(f"Model not loaded, loading model...")
			if self.load_model():
				print(f"Failed to load model, cannot get cell value.")
				return -1
		
		cell_tensor = self.preprocess(cell).unsqueeze(0)  # Add batch dimension

		with torch.no_grad():
			output = self.model(cell_tensor)
			_, predicted = torch.max(output, 1)
			cell_value = predicted.item()+1

		return cell_value

	def show_cells(self, n=9):
		"""
		Displays the first n cells of the Sudoku puzzle.
		For sanity checking.

		Args:
			n (int): The number of cells to display. Defaults to 9.
		"""
		fig_dim = math.sqrt(n)
		fig_dim = int(fig_dim) if fig_dim.is_integer() else int(fig_dim) + 1
		for i in range(min(n, len(self.cells))):
			plt.subplot(fig_dim, fig_dim, i + 1)
			plt.imshow(self.cells[i], cmap='gray')
			plt.axis('off')
		plt.show()
	
	def save_cells(self, path='data/img/numbers/'):
		"""
		Saves the individual cells of the Sudoku puzzle as image files.
		For Model training purposes.

		Args:
			path (str): The directory path where the cell images will be saved.
		"""
		if not os.path.exists(path):
			os.makedirs(path)
		for i, cell in enumerate(self.cells):
			cv2.imwrite(os.path.join(path, f'cell_{i}.png'), cell)

	def split_image(self):
		"""
		Method for splitting the Sudoku image into individual cells.
		
		Returns:
			list: A list of images representing individual Sudoku cells.
		"""
		# This is a placeholder implementation
		# Actual implementation would involve splitting the image into 81 cells
		cell_height = self.image.shape[0] // 9
		cell_width = self.image.shape[1] // 9
		cells = []
		for i in range(9):
			for j in range(9):
				cell = self.image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
				cells.append(cell)
		self.cells = cells

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


def isEmptyCell(cell):
	"""
	Checks if a Sudoku cell is empty based on mean pixel intensity in the center region.
	"""
	img = cell.copy()
	img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV, img)[1]

	# Gets the average pixel intensity of the center portion of the cell,
	# if it's over a certain threshold, we consider it a numbered cell.
	cell_intensity = np.mean(img[10:30, 10:30])
	return cell_intensity < CELL_POPULATED_THRESH

def puzzle_build_vis():
	"""
	Facade function to build a Sudoku puzzle from an image.
	Reads the image, detects corners, applies perspective transform, and identifies numbers in cells.
	
	Returns:
		sudoku_puzzle: An instance of the sudoku_puzzle class with identified numbers.
	"""
	try:
		sudoku_image = read_image()
		print("Image read successfully.")
		print("Image shape:", sudoku_image.shape)
	except FileNotFoundError as e:
		print(e)
	sudoku = puzzle_detector(sudoku_image)
	sudoku.preprocess_image()

	# Detect corners using Canny edge detection
	img, corners = sudoku.detect_corners_canny(1)

	# Use corners for perspective transform
	sudoku.perspective_transform(corners, show_flag=False)
	number_id = number_identifier(sudoku.image)

	puzzle = sudoku_puzzle()
	for i, cell in enumerate(number_id.cells):
		row = i // 9
		col = i % 9
		if isEmptyCell(cell):
			continue
		predicted_value = number_id.get_cell_val(cell)
		puzzle.puzzle[row][col].number = predicted_value
		if row == col:
			cv2.imshow(f"Cell {i}, predicted:{predicted_value}", cell)
			cv2.waitKey(5000)
		print(f"Predicted Value = {predicted_value}")
	
	return puzzle

if __name__ == "__main__":
	puzzle = puzzle_build_vis()
	puzzle.print_puzzle()