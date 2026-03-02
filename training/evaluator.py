import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
import logging

class Evaluator:
    def __init__(self, model, evaluation_dir):
        self.model = model
        self.evaluation_dir = evaluation_dir
        self.logger = logging.getLogger("SiameseExperiment")
        # Ensure images are scaled to tensor format (0 to 1)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def evaluate_20_way_one_shot(self, trials=400):
        """
        Lake developed a 20-way within-alphabet classification task in which an alphabet is first chosen from among those reserved for the evaluation set, along with twenty characters taken uniformly at random.
        This constitutes a total of 400 one-shot learning trials, from which the classification accuracy is calculated.
        """
        self.model.eval()
        correct_predictions = 0

        # Get the 10 evaluation alphabets [cite: 212]
        alphabets = os.listdir(self.evaluation_dir)

        # Ensure model is on the right device
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for trial in range(trials):
                # 1. Choose an alphabet at random
                alphabet = random.choice(alphabets)
                alphabet_path = os.path.join(self.evaluation_dir, alphabet)
                all_characters = os.listdir(alphabet_path)

                # 2. Sample 20 characters (or max available if < 20)
                num_classes = min(20, len(all_characters))
                sampled_characters = random.sample(all_characters, num_classes)

                # 3. Pick the "true" character we want to test against
                true_character = random.choice(sampled_characters)
                true_char_path = os.path.join(alphabet_path, true_character)

                # 4. "Two of the twenty drawers are also selected" [cite: 244, 245]
                # Pick two different images of the true character (representing 2 drawers)
                images = os.listdir(true_char_path)
                img1_name, img2_name = random.sample(images, 2)

                # Load the test image (Drawer 1)
                test_img_path = os.path.join(true_char_path, img1_name)
                test_image = Image.open(test_img_path).convert('L')
                test_tensor = self.transform(test_image).unsqueeze(0).to(device) # Shape: (1, 1, 105, 105)

                #5. Build the support set (Drawer 2)
                support_set = []
                true_class_index = -1

                for i, char in enumerate(sampled_characters):
                    char_path = os.path.join(alphabet_path, char)
                    if char == true_character:
                        # Use the second drawer's image for the correct class
                        support_img_path = os.path.join(char_path, img2_name)
                        true_class_index = i
                    else:
                        # Pick a random drawer for the incorrect classes
                        char_images = os.listdir(char_path)
                        support_img_path = os.path.join(char_path, random.choice(char_images))

                    support_img = Image.open(support_img_path).convert('L')
                    support_tensor = self.transform(support_img)
                    support_set.append(support_tensor)  

                # Stack the 20 support images into a single batch
                support_set_tensor = torch.stack(support_set).to(device) # Shape: (20, 1, 105, 105)
                
                # Duplicate the test image to match the batch size
                test_image_batch = test_tensor.repeat(num_classes, 1, 1, 1) # Shape: (20, 1, 105, 105)

                # 6. Feedforward pass
                similarity_scores = self.model(test_image_batch, support_set_tensor)
                
                # 7. "predict the class corresponding to the maximum similarity." [cite: 242]
                predicted_class_index = torch.argmax(similarity_scores).item()
                
                if predicted_class_index == true_class_index:
                    correct_predictions += 1
        
        accuracy = (correct_predictions / trials) * 100.0
        return accuracy