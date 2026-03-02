import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import torchvision.transforms as transforms
from data.augmentation import AffineAugmenter

class OmniglotDataset(Dataset):
    def __init__(self, data_dir, alphabets_list, drawers_list, num_pairs=30000, transform=None):
        self.data_dir = data_dir
        self.num_pairs = num_pairs
        self.transform = transform
        self.drawers_list = drawers_list

        self.characters = []

        for alphabet in alphabets_list:
            alphabet_path = os.path.join(data_dir, alphabet)
            for character in os.listdir(alphabet_path):
                character_path = os.path.join(alphabet_path, character)
                self.characters.append(character_path)

    def __len__(self):
        # we put together three different data set sizes with 30,000, 90,000, and 150,000 training examples by sampling random same and different pairs
        return self.num_pairs
    
    def _get_drawer_images(self, char_path):
        """Helper to filter images by the assigned drawers."""
        all_images = os.listdir(char_path)
        all_images.sort() 
        return [all_images[i] for i in self.drawers_list if i < len(all_images)]
    
    def __getitem__(self, idx):
        # The verification model learns to identify input pairs according to the probability that they belong to the same class or different classes
        
        # 50% chance to generate a "same" pair (target = 1.0), 50% chance for a "different" pair (target = 0.0)
        is_same_class = random.random() > 0.5

        if is_same_class:
            # Pick one random character class, then pick two different images (drawers) from it
            char_path = random.choice(self.characters)
            images = self._get_drawer_images(char_path)

            # Ensure we have at least 2 images to sample from
            if len(images) < 2:
                images = os.listdir(char_path) # Fallback if drawers are missing
                
            img1_name, img2_name = random.sample(images, 2)
            
            img1_path = os.path.join(char_path, img1_name)
            img2_path = os.path.join(char_path, img2_name)
            target = torch.tensor([1.0], dtype=torch.float32)
        else:
            # Pick two entirely different character classes, then pick one image from each
            char1_path, char2_path = random.sample(self.characters, 2)
            
            images1 = self._get_drawer_images(char1_path)
            images2 = self._get_drawer_images(char2_path)

            img1_name = random.choice(images1)
            img2_name = random.choice(images2)
            
            img1_path = os.path.join(char1_path, img1_name)
            img2_path = os.path.join(char2_path, img2_name)
            target = torch.tensor([0.0], dtype=torch.float32)

        # Load images and convert to grayscale
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        # ToTensor converts PIL Image (0-255) to PyTorch FloatTensor (0.0-1.0)
        to_tensor = transforms.ToTensor()
        img1 = to_tensor(img1)
        img2 = to_tensor(img2)

        # Apply transformations (AffineAugmenter or basic ToTensor)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, target
            

class OmniglotDataLoader:
    def __init__(self, background_dir, evaluation_dir, batch_size=128):
        """
        The background set is used for developing a model by learning hyperparameters and feature mappings.
        Conversely, the evaluation set is used only to measure the one-shot classification performance.
        """
        self.background_dir = background_dir
        self.evaluation_dir = evaluation_dir
        self.batch_size = batch_size
        self.logger = logging.getLogger("SiameseExperiment")
        self.augmenter = AffineAugmenter()

    def get_verification_loaders(self, batch_size):
        """
        We set aside sixty percent of the total data for training: 30 alphabets out of 50 and 12 drawers out of 20.
        First, we created a validation set for verification with 10,000 example pairs taken from 10 alphabets and 4 additional drawers.
        """
        all_background_alphabets = sorted(os.listdir(self.background_dir))

        self.logger.info("Segmenting background data into 30 training alphabets and 10 validation alphabets.")
        
        # We set aside sixty percent of the total data for training: 30 alphabets out of 50 and 12 drawers out of 20
        train_alphabets = all_background_alphabets[:30]
        train_drawers = list(range(0, 12)) # Indices 0 through 11 represent the first 12 drawers

        # First, we created a validation set for verification with 10,000 example pairs taken from 10 alphabets and 4 additional drawers
        val_alphabets = all_background_alphabets[30:40]
        val_drawers = list(range(12, 16)) # Indices 12 through 15 represent the next 4 drawers

        self.logger.info(f"Training: {len(train_alphabets)} alphabets, 12 drawers.")
        self.logger.info(f"Validation: {len(val_alphabets)} alphabets, 4 drawers.")

        # Train Dataset (90k pairs with augmentation)
        train_dataset = OmniglotDataset(
            data_dir=self.background_dir,
            alphabets_list=train_alphabets,
            drawers_list=train_drawers,
            num_pairs=90000, 
            transform=self.augmenter
        )

        # Validation Dataset (10k pairs, NO augmentation)
        val_dataset = OmniglotDataset(
            data_dir=self.background_dir,
            alphabets_list=val_alphabets,
            drawers_list=val_drawers,
            num_pairs=10000,
            transform=None # Never augment validation data
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader