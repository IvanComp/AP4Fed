import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from PIL import Image

import torch
from torchvision import models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def load_squeezenet_weights(model, weights_filename):
    state_dict = torch.load(weights_filename, weights_only=True)
    # squeezenet 5 clients
    state_dict['classifier.1.weight'] = state_dict.pop('classifier.1.1.weight')
    state_dict['classifier.1.bias'] = state_dict.pop('classifier.1.1.bias')
    model.load_state_dict(state_dict)
    return model

def load_shufflenet_weights(model, weights_filename):
    state_dict = torch.load(weights_filename, weights_only=True)
    # shufflenet rename keys fc.1.weight and fc.1.bias to fc.weight and fc.bias
    state_dict['fc.weight'] = state_dict.pop('fc.1.weight')
    state_dict['fc.bias'] = state_dict.pop('fc.1.bias')
    model.load_state_dict(state_dict)
    return model

def define_save_filename(weights_filename, method, image_filename):
    directory = os.path.dirname(weights_filename)
    image = os.path.splitext(os.path.basename(image_filename))[0]
    round = os.path.splitext(os.path.basename(weights_filename))[0]
    return f"{directory}/{method}_images/{method}_{image}_{round}.jpg"


def run_cam(
    model: torch.nn.Module,
    target_layers: list,
    weights_filename: str,
    image_filename: str,
):
    # skip gradcam computation if the image already exists
    if os.path.isfile(define_save_filename(weights_filename, "gradcam", image_filename)):
        return

    # configure input image
    img = get_image(image_filename)

    img = np.float32(img) / 255

    img_tensor = preprocess_image(
        img,
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    ).to("cpu")

    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]
    img_explanation = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    plt.imsave(define_save_filename(weights_filename, "gradcam", image_filename), img_explanation)
    return


def main(
    round: int = 1,
    model_weights_folder = "models/0-NoPatterns/shufflenet_v2_x0_5/model_weights/",
    model_name: str = "shufflenet_v2_x0_5",
    # model_name: str = "squeezenet_1_1",
):
    if model_name == "squeezenet_1_1":
        model = models.squeezenet1_1(weights=None, num_classes=10)
        target_layers = [model.features[12].expand3x3]
    elif model_name == "shufflenet_v2_x0_5":
        model = models.shufflenet_v2_x0_5(weights=None, num_classes=10)
        target_layers = [model.conv5[0]]

    # get image file paths
    images = glob.glob("images/val.X/**/*.JPEG")

    # compute the folder path for the model weights
    server_pt = f"{model_weights_folder}/server/MW_round{round}.pt"
    clients_pt = glob.glob(f"{model_weights_folder}/clients/*/MW_round{round}.pt")

    # create directories for gradcam images if they don't exist
    if not os.path.exists(f"{os.path.dirname(server_pt)}/gradcam_images"):
        os.makedirs(f"{os.path.dirname(server_pt)}/gradcam_images")

    for client_pt in clients_pt:
        if not os.path.exists(f"{os.path.dirname(client_pt)}/gradcam_images"):
            os.makedirs(f"{os.path.dirname(client_pt)}/gradcam_images")

    # load server model weights
    if model_name == "squeezenet_1_1":
        model = load_squeezenet_weights(model, server_pt)
    elif model_name == "shufflenet_v2_x0_5":
        model = load_shufflenet_weights(model, server_pt)

    # generate gradcam images for server
    for image in images:
        run_cam(
            model=model,
            target_layers=target_layers,
            weights_filename=server_pt,
            image_filename=image,
        )

    # generate gradcam images for each client
    for client_pt in clients_pt:
        if model_name == "squeezenet_1_1":
            model = load_squeezenet_weights(model, client_pt)
        elif model_name == "shufflenet_v2_x0_5":
            model = load_shufflenet_weights(model, client_pt)

        for image in images:
            run_cam(
                model=model,
                target_layers=target_layers,
                weights_filename=client_pt,
                image_filename=image,
            )

    # compute SSIM values
    ssim_values = []

    for client_pt in clients_pt:
        client_ssim = []
        for image in images:
            # compare server and client images
            server_image = get_image(define_save_filename(server_pt, "gradcam", image))
            client_image = get_image(define_save_filename(client_pt, "gradcam", image))

            server_image = img_as_float(server_image)
            client_image = img_as_float(client_image)

            ssim_value = ssim(server_image, client_image, data_range=server_image.max() - server_image.min(), channel_axis=2)
            client_ssim.append(ssim_value)

        client_ssim = np.array(client_ssim)
        ssim_values.append(client_ssim.mean())

    return ssim_values


import time
if __name__ == "__main__":
    start_time = time.time()
    ssim_values = main()
    print(ssim_values)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
