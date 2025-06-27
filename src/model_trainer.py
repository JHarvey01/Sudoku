import vis_reader as vr
import cv2
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 
from torchvision.models import resnet18	# Shouldn't need many layers.
import torch.nn as nn
import torch.optim as optim
import datetime

VALID_CELL_THRESH = 20  # Threshold to determine if a cell is valid (not empty) number of pixels

cell_counter = 0

MODEL_TEST_PATH = 'data/NumberDetection/test'
MODEL_TRAIN_PATH = 'data/NumberDetection/train'
MODEL_SAVE_PATH = 'data/models/resnet18/'

# Set the device to GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model_Trainer:
	"""
	A class to handle the training of a model for Sudoku number detection.
	
	Attributes:
		model (torch.nn.Module): The neural network model to be trained.
		train_loader (DataLoader): DataLoader for the training dataset.
		test_loader (DataLoader): DataLoader for the testing dataset.
	"""
	
	# Define the transformations to be applied to the images
	# from: https://docs.pytorch.org/vision/stable/transforms.html#start-here
	transforms = v2.Compose([
		v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
		v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
		v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	
	def __init__(self):
		self.model = resnet18(weights=None)  # Initialize a ResNet18 model
		self.model.fc = nn.Linear(self.model.fc.in_features, 10)
		self.train_loader, self.test_loader = self.get_data_loaders()
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(self.model.parameters(), lr=0.001)

		self.model.to(DEVICE)

		for epoch in range(5):  # Example: train for 10 epochs
			train_loss = self.train(criterion, optimizer)
			test_loss, accuracy = self.test(criterion)
			print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
		
				# Get current date and time
		timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')


		save_name = f'number_detection_model_{timestamp}.pth'
		torch.save(self.model.state_dict(), MODEL_SAVE_PATH+save_name)

	
	def get_data_loaders(self):
		"""		
		Loads the training and testing datasets for number detection.
		
		Returns:
			tuple: A tuple containing the training and testing DataLoaders.
		"""
		train_data = datasets.ImageFolder(MODEL_TRAIN_PATH, transform=self.transforms)
		test_data = datasets.ImageFolder(MODEL_TEST_PATH, transform=self.transforms)

		train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
		test_loader = DataLoader(test_data, batch_size=32)
		
		return train_loader, test_loader
	

	def train(self, criterion, optimizer):
		self.model.train()
		total_loss = 0

		for images, labels in self.train_loader:
				images, labels = images.to(DEVICE), labels.to(DEVICE)

				optimizer.zero_grad()
				outputs = self.model(images)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				total_loss += loss.item()
		
		return total_loss / len(self.train_loader)

	def test(self, criterion):
		self.model.eval()
		total_loss = 0
		correct = 0

		with torch.no_grad():
			for images, labels in self.test_loader:
				images, labels = images.to(DEVICE), labels.to(DEVICE)

				outputs = self.model(images)
				loss = criterion(outputs, labels)
				total_loss += loss.item()

				_, predicted = torch.max(outputs, 1)
				correct += (predicted == labels).sum().item()

		accuracy = correct / len(self.test_loader.dataset)
		return total_loss / len(self.test_loader), accuracy
	
def get_image_list():
	"""
	Returns a list of image file paths from the 'data/img' directory.
	
	Returns:
		list: A list of image file paths.
	"""
	img_dir = 'data/img/sudoku/'
	images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
	return images

def save_cell_images(cell, path='data/img/numbers/', verbose_flag=False):
	"""
	Saves the individual cells of the Sudoku puzzle as image files.
	
	Args:
		sudoku (vr.number_identifier): The number identifier object containing the Sudoku cells.
		path (str): The directory path where the cell images will be saved.
	"""
	if not os.path.exists(path):
		os.makedirs(path)
	
	global cell_counter

	img = cell.copy()
	img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV, img)[1]

	# Gets the average pixel intensity of the center portion of the cell,
	# if it's over a certain threshold, we consider it a numbered cell.
	cell_intensity = np.mean(img[10:30, 10:30])
	if verbose_flag:
		print(f"Cell: {cell_counter}, Intensity: {cell_intensity}")
	
	if cell_intensity < VALID_CELL_THRESH:
		if verbose_flag:
			print(f"Cell {cell_counter} is empty, skipping save.")
	else:
		# Save each cell image with a unique name
		cv2.imwrite(os.path.join(path, f'cell_{cell_counter}.png'), cell)
	cell_counter += 1
	return

if __name__ == "__main__":
	model_trainer = Model_Trainer()
	
	# img_list = get_image_list()
	# for img_path in img_list:
	# 	print(f"Processing image: {img_path}")
	# 	sudoku_image = vr.read_image(img_path)
	# 	sudoku = vr.puzzle_detector(sudoku_image)
	# 	sudoku.preprocess_image()
	# 	img, corners = sudoku.detect_corners_canny()
	
	# 	# Use corners for perspective transform
	# 	sudoku.perspective_transform(corners)
	# 	number_id = vr.number_identifier(sudoku.image)
	# 	for cell in number_id.cells:
	# 		save_cell_images(cell, path='data/img/numbers/')
		
