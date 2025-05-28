from typing import Any, Dict, Optional, Sequence, Tuple, Union

from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop

r"""OpenAI color normalization mean in RGB format (values in 0-1)."""
r"""OpenAI color normalization std in RGB format (values in 0-1)."""
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
r"""ImageNet color normalization mean in RGB format (values in 0-1)."""
r"""ImageNet color normalization std in RGB format (values in 0-1)."""
IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)


def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None
):
    mean = mean or IMAGENET_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or IMAGENET_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    normalize = Normalize(mean=mean, std=std)

    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)
