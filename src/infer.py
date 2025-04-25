import argparse

import cv2
import numpy as np
import torch
import yaml
from numpy.typing import NDArray

IMAGE_PATH = "../data/train-jpg/"
MAX_UINT8 = 255


def preprocess_image(image: NDArray, target_image_size: tuple[int, int]) -> torch.Tensor:
    image = image.astype(np.float32)
    image = cv2.resize(image, target_image_size) / MAX_UINT8
    image = np.transpose(image, (2, 0, 1))
    image -= np.array([0.485, 0.456, 0.406])[:, None, None]
    image /= np.array([0.229, 0.224, 0.225])[:, None, None]
    return torch.from_numpy(image)


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--model-path",
        default="../weights/model_jit.pt",
        metavar="MODEL PATH",
    )
    parser.add_argument(
        "-i",
        "--input-image",
        metavar="IMAGE PATH",
    )


if __name__ == "__main__":
    with open("../configs/labels.yaml", "r") as fin:
        labels = yaml.safe_load(fin)["labels"]
    parser = argparse.ArgumentParser(description="Model inference script")
    setup_parser(parser)
    args = parser.parse_args()
    model = torch.jit.load(args.model_path, map_location="cpu")
    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_image(img, (224, 224))

    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(img[None])).detach().cpu().numpy()[0]
        probs = dict(zip(labels, probs))
        print(probs)
