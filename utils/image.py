from rembg import remove
import torch
from typing import Union, Callable
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import rotate
import os


def preprocess_image(path_or_data: Union[str, Image.Image],
                     preprocess_func: Callable,
                     path_to_save: str=None) -> torch.Tensor:
    """"""
    input = path_or_data
    if isinstance(path_or_data, str):
        if os.path.isfile(path_or_data):
            input = Image.open(path_or_data)

    output = preprocess_func(input)

    if isinstance(output, Image.Image):
        output = transforms.ToTensor()(output)

    if path_to_save:
            transforms.ToPILImage()(output).save(path_to_save)

    return output


def remove_background_of_image(path_or_data: Union[str, Image.Image],
                               path_to_save: str=None) -> torch.Tensor:
    """"""
    return preprocess_image(path_or_data=path_or_data,
                            preprocess_func=remove,
                            path_to_save=path_to_save)


def rotate_ccw_45(path_or_data: Union[str, Image.Image],
              path_to_save: str=None) -> torch.Tensor:
    """"""
    def _rotate_ccw_45(img):
        return rotate(img, 45)

    return preprocess_image(path_or_data=path_or_data,
                        preprocess_func=_rotate_ccw_45,
                        path_to_save=path_to_save)


def rotate_ccw_90(path_or_data: Union[str, Image.Image],
              path_to_save: str=None) -> torch.Tensor:
    """"""
    def _rotate_ccw_90(img):
        return rotate(img, 90)

    return preprocess_image(path_or_data=path_or_data,
                        preprocess_func=_rotate_ccw_90,
                        path_to_save=path_to_save)


def rotate_ccw_135(path_or_data: Union[str, Image.Image],
              path_to_save: str=None) -> torch.Tensor:
    """"""
    def _rotate_ccw_135(img):
        return rotate(img, 135)

    return preprocess_image(path_or_data=path_or_data,
                        preprocess_func=_rotate_ccw_135,
                        path_to_save=path_to_save)


def rotate_180(path_or_data: Union[str, Image.Image],
              path_to_save: str=None) -> torch.Tensor:
    """"""
    def _rotate_180(img):
        return rotate(img, 180)

    return preprocess_image(path_or_data=path_or_data,
                        preprocess_func=_rotate_180,
                        path_to_save=path_to_save)


def rotate_cw_45(path_or_data: Union[str, Image.Image],
              path_to_save: str=None) -> torch.Tensor:
    """"""
    def _rotate_cw_45(img):
        return rotate(img, -45)

    return preprocess_image(path_or_data=path_or_data,
                        preprocess_func=_rotate_cw_45,
                        path_to_save=path_to_save)


def rotate_cw_135(path_or_data: Union[str, Image.Image],
              path_to_save: str=None) -> torch.Tensor:
    """"""
    def _rotate_cw_135(img):
        return rotate(img, -135)

    return preprocess_image(path_or_data=path_or_data,
                        preprocess_func=_rotate_cw_135,
                        path_to_save=path_to_save)


def rotate_cw_90(path_or_data: Union[str, Image.Image],
              path_to_save: str=None) -> torch.Tensor:
    """"""
    def _rotate_cw_90(img):
        return rotate(img, -90)

    return preprocess_image(path_or_data=path_or_data,
                        preprocess_func=_rotate_cw_90,
                        path_to_save=path_to_save)