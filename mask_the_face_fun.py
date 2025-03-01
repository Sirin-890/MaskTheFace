
import dlib
import os
import numpy as np
import cv2
from tqdm import tqdm
from utils.aux_functions import check_path, is_image, display_MaskTheFace, download_dlib_model

def mask_the_face(
    path="",
    mask_type="surgical",
    pattern="",
    pattern_weight=0.5,
    color="#0473e2",
    color_weight=0.5,
    code="",
    verbose=False,
    write_original_image=False
):
    """
    MaskTheFace - Function to mask faces dataset
    
    Parameters:
    path (str): Path to either the folder containing images or the image itself
    mask_type (str): Type of the mask to be applied. Available options: all, surgical, N95, KN95, cloth, gas, inpaint, random
    pattern (str): Type of the pattern. Available options in masks/textures
    pattern_weight (float): Weight of the pattern. Must be between 0 and 1
    color (str): Hex color value that need to be overlayed to the mask
    color_weight (float): Weight of the color intensity. Must be between 0 and 1
    code (str): Generate specific formats
    verbose (bool): Turn verbosity on
    write_original_image (bool): If true, original image is also stored in the masked folder
    
    Returns:
    str: Path to the masked images
    """
    
    # Create the write path
    write_path = path + "_masked"
    
    # Set up dlib face detector and predictor
    detector = dlib.get_frontal_face_detector()
    path_to_dlib_model = "dlib_models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(path_to_dlib_model):
        download_dlib_model()
    
    predictor = dlib.shape_predictor(path_to_dlib_model)
    
    # Extract data from code
    mask_dict_of_dict = {}
    code_count = None
    
    if code:
        mask_code = "".join(code.split()).split(",")
        code_count = np.zeros(len(mask_code))
        
        for i, entry in enumerate(mask_code):
            mask_dict = {}
            mask_color = ""
            mask_texture = ""
            mask_type_from_code = entry.split("-")[0]
            if len(entry.split("-")) == 2:
                mask_variation = entry.split("-")[1]
                if "#" in mask_variation:
                    mask_color = mask_variation
                else:
                    mask_texture = mask_variation
            mask_dict["type"] = mask_type_from_code
            mask_dict["color"] = mask_color
            mask_dict["texture"] = mask_texture
            mask_dict_of_dict[i] = mask_dict
    
    # Check if path is file or directory or none
    is_directory, is_file, is_other = check_path(path)
    display_MaskTheFace()
    
    # Modified version of mask_image that works without args
    def modified_mask_image(image_path):
        """Apply mask to a single image without using args"""
        # Import the mask_image function dynamically to avoid circular imports
        from utils.aux_functions import mask_image as original_mask_image
        
        # Create a temporary container with required attributes for compatibility
        class TempContainer:
            pass
        
        container = TempContainer()
        container.detector = detector
        container.predictor = predictor
        container.mask_type = mask_type
        container.pattern = pattern
        container.pattern_weight = pattern_weight
        container.color = color
        container.color_weight = color_weight
        container.code = code
        container.code_count = code_count
        container.mask_dict_of_dict = mask_dict_of_dict
        
        return original_mask_image(image_path, container)
    
    if is_directory:
        path_dir, dirs, files = os.walk(path).__next__()
        file_count = len(files)
        dirs_count = len(dirs)
        if len(files) > 0:
            print_orderly("Masking image files", 60)
    
        # Process files in the directory if any
        for f in tqdm(files):
            image_path = path_dir + "/" + f
    
            if not os.path.isdir(write_path):
                os.makedirs(write_path)
    
            if is_image(image_path):
                # Proceed if file is image
                if verbose:
                    str_p = "Processing: " + image_path
                    tqdm.write(str_p)
    
                split_path = f.rsplit(".")
                masked_image, mask, mask_binary_array, original_image = modified_mask_image(image_path)
                
                for i in range(len(mask)):
                    w_path = (
                        write_path
                        + "/"
                        + split_path[0]
                        + "_"
                        + mask[i]
                        + "."
                        + split_path[1]
                    )
                    img = masked_image[i]
                    cv2.imwrite(w_path, img)
    
        print_orderly("Masking image directories", 60)
    
        # Process directories within the path provided
        for d in tqdm(dirs):
            dir_path = path + "/" + d
            dir_write_path = write_path + "/" + d
            if not os.path.isdir(dir_write_path):
                os.makedirs(dir_write_path)
            _, _, files = os.walk(dir_path).__next__()
    
            # Process each file within subdirectory
            for f in files:
                image_path = dir_path + "/" + f
                if verbose:
                    str_p = "Processing: " + image_path
                    tqdm.write(str_p)
                
                if is_image(image_path):
                    # Proceed if file is image
                    split_path = f.rsplit(".")
                    masked_image, mask, mask_binary, original_image = modified_mask_image(image_path)
                    
                    for i in range(len(mask)):
                        w_path = (
                            dir_write_path
                            + "/"
                            + split_path[0]
                            + "_"
                            + mask[i]
                            + "."
                            + split_path[1]
                        )
                        w_path_original = dir_write_path + "/" + f
                        img = masked_image[i]
                        # Write the masked image
                        cv2.imwrite(w_path, img)
                        if write_original_image:
                            # Write the original image
                            cv2.imwrite(w_path_original, original_image)
    
                if verbose and code_count is not None:
                    print(code_count)
    
    # Process if the path was a file
    elif is_file:
        print("Masking image file")
        image_path = path
        write_path_base = path.rsplit(".")[0]
        if is_image(image_path):
            # Proceed if file is image
            masked_image, mask, mask_binary_array, original_image = modified_mask_image(image_path)
            
            for i in range(len(mask)):
                w_path = write_path_base + "_" + mask[i] + "." + path.rsplit(".")[1]
                img = masked_image[i]
                cv2.imwrite(w_path, img)
    else:
        print("Path is neither a valid file or a valid directory")
    
    print("Processing Done")
    return img

# Helper function that was used in the original code
def print_orderly(text, num=80):
    """Helper function to print messages in an orderly fashion"""
    print("=" * num)
    print(text)
    print("=" * num)