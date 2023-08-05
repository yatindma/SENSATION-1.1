import logging
import shutil
import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from data import Dataset, get_preprocessing, CLASSES
from model import Trained_model, decode_segmentation_map, preprocessing_fn
from utils import get_input_file_name, resize_video, generate_frames, combine_predicted_outputs

# Set up logging
logging.basicConfig(filename='application.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Check if CUDA is available, otherwise use CPU (mps)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


def main():
    """
    Main function to process the video and generate segmentation output.

    Steps:
    1. Get input video file name from user.
    2. Resize the video frames.
    3. Generate individual frames from the resized video.
    4. Generate predicted segmentation masks for each frame.
    5. Combine the predicted outputs to create a segmented video.
    6. Log the time taken for segmentation.
    """
    try:
        input_file_Name = get_input_file_name()
        logging.info('Final_Folder/Input_Video/' + input_file_Name)

        start_time1 = time.time()

        resize_video(input_file_Name)
        logging.info("Resizing done")

        generate_frames()
        logging.info("Frames generated")

        generate_segmentation_output()
        logging.info("Predicted Segmentation Generated")

        combine_predicted_outputs(input_file_Name)

        end_time1 = time.time()
        logging.info("Video generated")
        logging.info(f"Time taken to segment this video: {round(end_time1 - start_time1, 2)} seconds")
    except Exception as e:
        # cleanup()
        logging.error(f"Video not generated due to error: {e}")


def generate_segmentation_output():
    """
    Generate segmentation output for each frame in the video.

    Steps:
    1. Load the test dataset containing frames and corresponding ground truth masks.
    2. Process each frame through the trained model to get the predicted segmentation mask.
    3. Convert the predicted mask to a color-coded RGB image.
    4. Save the segmented frame as an image.
    5. Log the time taken for segmentation.
    """
    try:
        x_test_dir = 'Final_Folder/Original_Frames'
        y_test_dir = 'Final_Folder/Original_Frames'
        output_path = 'Final_Folder/Predicted_Output'

        # Create the test dataset
        test_dataset = Dataset(
            x_test_dir,
            y_test_dir,
            augmentation=None,
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )

        start_time = time.time()
        for j in range(len(test_dataset)):
            image_filename, gt_mask_filename = test_dataset.images_fps[j], test_dataset.masks_fps[j]

            # Read the image files into memory as numpy arrays
            image = cv2.imread(image_filename)
            gt_mask = cv2.imread(gt_mask_filename)

            # If the image is read as None, skip this iteration
            if image is None or gt_mask is None:
                continue

            # Convert the image data into PyTorch tensors
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = preprocessing_fn(image)  # Preprocess the image
            image = np.transpose(image, (2, 0, 1))  # Change data from HxWxC to CxHxW
            x_tensor = torch.from_numpy(image).float().to(DEVICE).unsqueeze(
                0)  # Convert to torch tensor and add batch dimension

            with torch.no_grad():
                # Forward pass to get the predicted mask
                predicted_mask = Trained_model.module.predict(x_tensor)

            # Decode the predicted mask into a color-coded RGB image
            predicted_output = torch.argmax(predicted_mask.squeeze(), dim=0).detach().cpu().numpy()
            rgb_map = decode_segmentation_map(predicted_output)

            # Save the segmented frame as an image
            fig, ax = plt.subplots()
            ax.imshow(rgb_map, alpha=0.5)
            ax.axis('off')
            fig.savefig(f"{output_path}/{str(j)}.jpg")

        logging.info(f"Time taken for segmentation output : {time.time() - start_time}")

    except Exception as e:
        logging.error(f"Error occurred while generating segmentation output: {e}")


def cleanup():
    """
    Clean up the final folder after video generation (Not included in the current flow).
    """
    dir_path = 'Final_Folder/'
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        logging.error(f"Error in cleanup: {e.filename} - {e.strerror}")


if __name__ == "__main__":
    main()
