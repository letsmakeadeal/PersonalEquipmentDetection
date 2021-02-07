import numpy as np
import random

import albumentations as AT

__all__ = ['ResizeWithKeepAspectRatio', 'AugMix']


class ResizeWithKeepAspectRatio(object):
    def __init__(self,
                 height: int,
                 width: int,
                 divider: int):
        self._height = height
        self._width = width
        self._resize_if_width_bigger = AT.Compose([
            AT.LongestMaxSize(max_size=width),
            AT.PadIfNeeded(min_width=(width // divider) * divider, min_height=(height // divider) * divider,
                           value=(0, 0, 0), border_mode=0),
            AT.CenterCrop(width=(width // divider) * divider, height=(height // divider) * divider)

        ])
        self._resize_if_height_bigger = AT.Compose([
            AT.LongestMaxSize(max_size=height),
            AT.PadIfNeeded(min_width=(width // divider) * divider, min_height=(height // divider) * divider,
                           value=(0, 0, 0), border_mode=0),
            AT.CenterCrop(width=(width // divider) * divider, height=(height // divider) * divider)
        ])

    def __call__(self, force_apply=False, **data):
        h, w, _ = data['image'].shape
        output = self._resize_if_height_bigger(**data) if h > w else self._resize_if_width_bigger(**data)
        return output


class AugMix(object):
    """Perform AugMix augmentations and compute mixture.
    Code taken and adapted from https://arxiv.org/pdf/1912.02781.pdf
    Args:
      severity: Severity of underlying augmentation operators (between 1 to 10).
      width: Width of augmentation chain
      depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.
    """

    def __init__(self,
                 severity: int = 3,
                 width: int = 3,
                 depth: int = -1,
                 alpha: float = 1.,
                 geometric_aug_level: float = 0.2,
                 p: float = 1):
        self._severity = severity
        self._width = width
        self._depth = depth
        self._alpha = alpha
        self._geometric_aug_level = geometric_aug_level
        self._p = p

    @staticmethod
    def _int_parameter(level, maxval):
        return int(level * maxval / 10)

    @staticmethod
    def _float_parameter(level, maxval):
        return float(level) * maxval / 10.

    @staticmethod
    def _sample_level(n):
        return np.random.uniform(low=0.1, high=n)

    def _sample_geometric_level(self, n):
        return np.random.uniform(low=0., high=self._geometric_aug_level * n)

    def _get_augmentations(self):
        from PIL import ImageOps

        def _autocontrast(pil_img, _):
            return ImageOps.autocontrast(pil_img)

        def _equalize(pil_img, _):
            return ImageOps.equalize(pil_img)

        def _posterize(pil_img, level):
            level = AugMix._int_parameter(AugMix._sample_level(level), 4)
            return ImageOps.posterize(pil_img, 4 - level)

        def _solarize(pil_img, level):
            level = AugMix._int_parameter(AugMix._sample_level(level), 256)
            return ImageOps.solarize(pil_img, 256 - level)

        return [
            _autocontrast, _equalize, _posterize, _solarize,
        ]

    def __call__(self, force_apply=False, **data):
        if random.random() > self._p:
            return data

        assert data['image'].dtype == np.uint8

        def _make_aug_img(original_img):
            def _apply_op(image, op, severity):
                from PIL import Image
                image = np.clip(image * 255., 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(image)
                pil_img = op(pil_img, severity)
                return np.asarray(pil_img) / 255.

            original_img = original_img / 255.
            ws = np.float32(np.random.dirichlet([self._alpha] * self._width))
            m = np.float32(np.random.beta(self._alpha, self._alpha))
            augmentations = self._get_augmentations()

            mix = np.zeros_like(original_img)
            for i in range(self._width):
                image_aug = original_img.copy()
                depth = self._depth if self._depth > 0 else np.random.randint(1, 4)
                for _ in range(depth):
                    op = np.random.choice(augmentations)
                    image_aug = _apply_op(image_aug, op, self._severity)
                mix += ws[i] * image_aug

            return (((1 - m) * original_img + m * mix) * 255.).astype(np.uint8)

        data['image'] = _make_aug_img(data['image'])

        return data
