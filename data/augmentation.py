import random
import torchvision.transforms.functional as TF
from PIL import Image

class AffineAugmenter:
    def __init__(self):
        self.theta_range = (-10.0, 10.0)
        self.shear_range = (-0.3 * 180 / 3.14159, 0.3 * 180 / 3.14159)
        self.scale_range = (0.8, 1.2)
        self.translate_range = (-2, 2)

    def _get_random_param(self, param_range):
        """Draws stochastically from a multidimensional uniform distribution."""
        return random.uniform(param_range[0], param_range[1])

    def __call__(self, img):
        """
        Applies the affine transformation to a single PIL Image.
        Each of these components of the transformation is included with probability 0.5.
        """
        angle = self._get_random_param(self.theta_range) if random.random() > 0.5 else 0.0
        translate_x = int(self._get_random_param(self.translate_range)) if random.random() > 0.5 else 0
        translate_y = int(self._get_random_param(self.translate_range)) if random.random() > 0.5 else 0
        translations = (translate_x, translate_y)
        scale = self._get_random_param(self.scale_range) if random.random() > 0.5 else 1.0
        shear_x = self._get_random_param(self.shear_range) if random.random() > 0.5 else 0.0
        shear_y = self._get_random_param(self.shear_range) if random.random() > 0.5 else 0.0
        shears = (shear_x, shear_y)

        augmented_img = TF.affine(
            img, 
            angle=angle, 
            translate=translations, 
            scale=scale, 
            shear=shears,
            fill=0 # Assuming background is black/0
        )
        return augmented_img