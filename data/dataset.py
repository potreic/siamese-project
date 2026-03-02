import torch
from torch.utils.data import Dataset, DataLoader
import logging

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
    
    
