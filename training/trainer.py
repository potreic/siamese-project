import torch
import torch.nn as nn
import torch.optim as optim
import logging

class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logging.getLogger(__name__)

        # We impose a regularized cross-entropy objective on our binary classifier
        self.criterion = nn.BCELoss()

        self.initial_lr = 0.01 
        self.l2_penalty = 0.05

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr, momentum=0.5, weight_decay=self.l2_penalty)

        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

    def train(self, max_epochs=200):
        best_val_error = float('inf')
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            running_loss = 0.0

            # Simulated training loop over minibatches
            # for x1, x2, y in self.train_loader:
            #     self.optimizer.zero_grad()
            #     outputs = self.model(x1, x2)
            #     loss = self.criterion(outputs, y)
            #     loss.backward()
            #     self.optimizer.step()
            #     running_loss += loss.item()

            # Step the scheduler
            self.scheduler.step()
            
            # TODO: Implement linear momentum scaling per epoch here
            
            # Validation phase
            self.model.eval()
            
            # monitored one-shot validation error on a set of 320 one-shot learning tasks generated randomly from the alphabets and drawers in the validation set
            val_error = self._evaluate_validation_one_shot(tasks=320)

            self.logger.info(f"[Epoch {epoch:03d}/{max_epochs}] Loss: {running_loss:.4f} | Val Error: {val_error:.2f}%")

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
        # TODO: Implement the 320-task generation and evaluation
        # Returns simulated error percentage for boilerplate
        return 10.5