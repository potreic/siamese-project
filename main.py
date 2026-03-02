import logging
import os
import torch

# Import from custom modules
from utils.logger import setup_logger
from data.dataset import OmniglotDataLoader
from models.siamese_net import SiameseNetwork
from training.trainer import Trainer
from training.evaluator import Evaluator

def main():
    # Initialize neat formatted logger
    logger = setup_logger(log_file="experiment.log")
    logger.info("=== SYSTEM INITIALIZATION ===")

    # Path
    bg_path = r"D:\Skripsi\playground\data\omniglot-py\images_background"
    eval_path = r"D:\Skripsi\playground\data\omniglot-py\images_evaluation"

    logger.info(f"Background Data Path: {bg_path}")
    logger.info(f"Evaluation Data Path: {eval_path}")

    # Data prep and split
    logger.info("=== DATA PREPARATION & SPLITTING ===")
    # split the data into a 40 alphabet background set and a 10 alphabet evaluation set
    data_loader = OmniglotDataLoader(background_dir=bg_path, evaluation_dir=eval_path, batch_size=128)
    
    train_loader = data_loader.get_train_loader(num_pairs=90000)
    logger.info("Successfully initialized DataLoaders with affine augmentation.")

    logger.info("=== MODEL & HYPERPARAMETERS INITIALIZATION ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = SiameseNetwork().to(device)
    logger.info("Siamese Convolutional Neural Network instantiated.")
    
    logger.info("=== TRAINING LOOP STARTED ===")
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=None)

    # We trained each network for a maximum of 200 epochs
    best_model_weights = trainer.train(max_epochs=200)
    model.load_state_dict(best_model_weights)
    
    # final evaluation
    logger.info("=== FINAL ONE-SHOT EVALUATION ===")
    evaluator = Evaluator(model=model, evaluation_dir=eval_path)
    
    # This constitutes a total of 400 one-shot learning trials, from which the classification accuracy is calculated.
    final_accuracy = evaluator.evaluate_20_way_one_shot(trials=400)

    logger.info(f"Final One-Shot Accuracy: {final_accuracy:.2f}%")
    logger.info("=== EXPERIMENT COMPLETE ===")

if __name__ == "__main__":
    main()