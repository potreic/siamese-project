import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from PIL import Image
import torchvision.transforms as transforms

class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logging.getLogger("SiameseExperiment")

        # We impose a regularized cross-entropy objective on our binary classifier
        self.criterion = nn.BCELoss()

        self.initial_lr = 0.01 
        self.l2_penalty = 0.05
        self.target_momentum = 0.9

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr, momentum=0.5, weight_decay=self.l2_penalty)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def train(self, max_epochs=200):
        best_val_error = float('inf')
        epochs_without_improvement = 0
        best_model_state = None
        device = next(self.model.parameters()).device

        momentum_increment = (self.target_momentum - 0.5) / 50.0

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            running_loss = 0.0

            # training loop over minibatches
            for x1, x2, y in self.train_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(x1, x2)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Step the scheduler
            self.scheduler.step()
            
            # increasing linearly each epoch until reaching the value mu_j, the individual momentum term for the jth layer.
            current_momentum = min(self.target_momentum, 0.5 + (epoch * momentum_increment))
            for param_group in self.optimizer.param_groups:
                param_group['momentum'] = current_momentum
            
            # Validation phase
            self.model.eval()
            
            # monitored one-shot validation error on a set of 320 one-shot learning tasks generated randomly from the alphabets and drawers in the validation set
            val_error = self._evaluate_validation_one_shot(tasks=320)

            #Average the loss over the number of batches
            avg_train_loss = running_loss / len(self.train_loader)
            self.logger.info(f"[Epoch {epoch:03d}/{max_epochs}] Loss: {avg_train_loss:.4f} | Val Error: {val_error:.2f}% | Momentum: {current_momentum:.3f}")
            # self.logger.info(f"[Epoch {epoch:03d}/{max_epochs}] Loss: {running_loss:.4f} | Val Error: {val_error:.2f}%")

            # When the validation error did not decrease for 20 epochs, we stopped and used the parameters of the model at the best epoch
            if val_error < best_val_error:
                best_val_error = val_error
                epochs_without_improvement = 0
                best_model_state = self.model.state_dict()
                self.logger.info("Validation error decreased. Saving best model state.")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 20:
                self.logger.info(f"Early stopping triggered at epoch {epoch}. No improvement for 20 epochs.")
                break
        
        self.model.load_state_dict(best_model_state)
        return best_model_state
    
    def _evaluate_validation_one_shot(self, tasks=320):
        correct = 0
        device = next(self.model.parameters()).device
        val_dataset = self.val_loader.dataset 
        to_tensor = transforms.ToTensor()

        # Group characters back into their alphabets using the dataset paths
        alphabets_dict = {}
        for char_path in val_dataset.characters:
            alphabet_name = os.path.basename(os.path.dirname(char_path))
            if alphabet_name not in alphabets_dict:
                alphabets_dict[alphabet_name] = []
            alphabets_dict[alphabet_name].append(char_path)
        
        with torch.no_grad():
            for _ in range(tasks):
                # 1. Choose an alphabet from the validation set
                alphabet = random.choice(list(alphabets_dict.keys()))
                chars_in_alphabet = alphabets_dict[alphabet]
                
                # 2. Sample 20 characters
                num_classes = min(20, len(chars_in_alphabet))
                sampled_chars = random.sample(chars_in_alphabet, num_classes)
                
                # 3. Pick the true character
                true_char_path = random.choice(sampled_chars)
                
                # 4. Pick 2 drawers exclusively from the validation set's drawers
                drawer1, drawer2 = random.sample(val_dataset.drawers_list, 2)
                
                # Load test image (Drawer 1)
                test_imgs = sorted(os.listdir(true_char_path))
                # Ensure the drawer index doesn't exceed available images
                d1_idx = drawer1 if drawer1 < len(test_imgs) else 0 
                test_img_path = os.path.join(true_char_path, test_imgs[d1_idx])
                test_img = to_tensor(Image.open(test_img_path).convert('L')).unsqueeze(0).to(device)
                
                # Load support set (Drawer 2)
                support_set = []
                true_class_index = -1
                
                for i, char_path in enumerate(sampled_chars):
                    if char_path == true_char_path:
                        true_class_index = i
                        
                    char_imgs = sorted(os.listdir(char_path))
                    d2_idx = drawer2 if drawer2 < len(char_imgs) else 0
                    supp_img_path = os.path.join(char_path, char_imgs[d2_idx])
                    
                    supp_img = to_tensor(Image.open(supp_img_path).convert('L')).to(device)
                    support_set.append(supp_img)
                    
                support_set_tensor = torch.stack(support_set)
                test_image_batch = test_img.repeat(num_classes, 1, 1, 1)
                
                # Feedforward pass
                scores = self.model(test_image_batch, support_set_tensor)
                pred = torch.argmax(scores).item()
                
                if pred == true_class_index:
                    correct += 1
                    
        # Calculate error instead of accuracy, since early stopping looks for a decrease
        error_rate = 100.0 - ((correct / tasks) * 100.0)
        return error_rate
    
        return 10.5