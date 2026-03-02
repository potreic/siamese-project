import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import torchvision.transforms as transforms

class OmniglotDataset(Dataset):
    def __init__(self, data_dir, num_pairs=30000, transform=None):
        self.data_dir = data_dir
        self.num_pairs = num_pairs
        self.transform = transform

        # Load the directory structure: alphabets -> characters -> images
        self.alphabets = os.listdir(data_dir)
        self.characters = []

        for alphabet in self.alphabets:
            alphabet_path = os.path.join(data_dir, alphabet)
            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)
                self.characters.append(character_path)

    def __len__(self):
        # we put together three different data set sizes with 30,000, 90,000, and 150,000 training examples by sampling random same and different pairs
        return self.num_pairs
    
    def __getitem__(self, idx):
        # The verification model learns to identify input pairs according to the probability that they belong to the same class or different classes
        
        # 50% chance to generate a "same" pair (target = 1.0), 50% chance for a "different" pair (target = 0.0)
        is_same_class = random.random() > 0.5

        if is_same_class:
            # Pick one random character class, then pick two different images (drawers) from it
            char_path = random.choice(self.characters)
            images = os.listdir(char_path)
            img1_name, img2_name = random.sample(images, 2)
            
            img1_path = os.path.join(char_path, img1_name)
            img2_path = os.path.join(char_path, img2_name)
            target = torch.tensor([1.0], dtype=torch.float32)
        else:
            # Pick two entirely different character classes, then pick one image from each
            char1_path, char2_path = random.sample(self.characters, 2)
            
            img1_name = random.choice(os.listdir(char1_path))
            img2_name = random.choice(os.listdir(char2_path))
            
            img1_path = os.path.join(char1_path, img1_name)
            img2_path = os.path.join(char2_path, img2_name)
            target = torch.tensor([0.0], dtype=torch.float32)

        # Load images and convert to grayscale
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        # Apply transformations (AffineAugmenter or basic ToTensor)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            # Ensure images are properly formatted as PyTorch tensors even without augmentation
            base_transform = transforms.Compose([
                transforms.ToTensor() 
            ])
            img1 = base_transform(img1)
            img2 = base_transform(img2)

        return img1, img2, target
            

class OmniglotDataLoader:
    def __init__(self, background_dir, evaluation_dir):
        """
        The background set is used for developing a model by learning hyperparameters and feature mappings.
        Conversely, the evaluation set is used only to measure the one-shot classification performance.
        """
        self.background_dir = background_dir
        self.evaluation_dir = evaluation_dir
        self.logger = logging.getLogger(__name__)

    def get_verification_loaders(self, batch_size):
        """
        We set aside sixty percent of the total data for training: 30 alphabets out of 50 and 12 drawers out of 20.
        First, we created a validation set for verification with 10,000 example pairs taken from 10 alphabets and 4 additional drawers.
        """
        self.logger.info("Segmenting background data into 30 training alphabets and 10 validation alphabets.")
        
        # TODO: Implement actual image loading and pairing logic here
        train_loader = None 
        val_loader = None

        return train_loader, val_loader
    

