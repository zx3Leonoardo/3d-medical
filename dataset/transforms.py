import random

class RandomFlip_lr:
    def __init__(self, prob) -> None:
        self.prob = prob

    def _flip(self, img, prob):
        if prob<=self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, label) -> Any:
        prob = random.uniform(0, 1)
        return self._flip(img, self.prob), self._flip(label, self.prob)

class RandomFlip_ud:
    def __init__(self, prob) -> None:
        self.prob = prob
    
    def _flip(self, img, prob):
        if prob<=self.prob:
            img = img.flip(3)
        return img

    def __calll__(self, img, label):
        prob = random.uniform(0, 1)
        return self._flip(img, self.prob), self._flip(label, self.prob)

class Compose:
    def __init__(self,transforms):
        self.transforms = transforms
    
    def __call__(self, img, label) -> Any:
        for tr in self.transforms:
            img, label = tr(img, label)
        return img, label