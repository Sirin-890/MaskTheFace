import dlib
import os
import cv2
import numpy as np
from utils.aux_functions import *

def mask_image_with_args(
    path, mask_type="surgical", pattern="", pattern_weight=0.5, color="#0473e2",
    color_weight=0.5, code="", verbose=False, write_original_image=False
):
    # Set up dlib face detector and predictor
    detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model()
    predictor = dlib.shape_predictor(path_to_dlib_model)

    # Extract data from code
    mask_code = "".join(code.split()).split(",")
    mask_dict_of_dict = {}
    for i, entry in enumerate(mask_code):
        mask_dict = {}
        mask_color = ""
        mask_texture = ""
        mask_type = entry.split("-")[0]
        if len(entry.split("-")) == 2:
            mask_variation = entry.split("-")[1]
            if "#" in mask_variation:
                mask_color = mask_variation
            else:
                mask_texture = mask_variation
        mask_dict["type"] = mask_type
        mask_dict["color"] = mask_color
        mask_dict["texture"] = mask_texture
        mask_dict_of_dict[i] = mask_dict

    # Check if path is file or directory or none
    is_directory, is_file, is_other = check_path(path)
    display_MaskTheFace()

    # Function to process images and return a masked image
    def process_image(image_path):
        args = {
            "mask_type": mask_type,
            "pattern": pattern,
            "pattern_weight": pattern_weight,
            "color": color,
            "color_weight": color_weight,
            "code": code,
            "verbose": verbose,
            "write_original_image": write_original_image,
            "write_path": path + "_masked",
            "detector": detector,
            "predictor": predictor,
            "mask_dict_of_dict": mask_dict_of_dict,
            "code_count": np.zeros(len(mask_code)),
        }

        masked_image, mask, mask_binary_array, original_image = mask_image(image_path, args)
        return masked_image[0]  # Return the first masked image

    if is_directory:
        path, dirs, files = os.walk(path).__next__()
        for f in files:
            image_path = path + "/" + f
            if is_image(image_path):
                if verbose:
                    str_p = "Processing: " + image_path
                    print(str_p)
                return process_image(image_path)

    elif is_file:
        if verbose:
            print("Masking image file")
        image_path = path
        if is_image(image_path):
            return process_image(image_path)
    else:
        if verbose:
            print("Path is neither a valid file nor a valid directory")

    if verbose:
        print("Processing Done")
    return None

