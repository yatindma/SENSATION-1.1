import os
import cv2

INPUT_FOLDER = 'Final_Folder/Input_Video/'
RESIZED_FOLDER = 'Final_Folder/Resized_Video/'
FRAMES_FOLDER = 'Final_Folder/Original_Frames/'
PREDICTED_OUTPUT_FOLDER = 'Final_Folder/Predicted_Output/'
FINAL_VIDEO_FOLDER = 'Final_Folder/Final_Video/'


def get_input_file_name():
    """
    Function to get the name of the input file.

    Returns:
        str: Name of the input file read from 'Input_File.txt'.
    """
    try:
        with open('Input_File.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print("Input_File.txt not found")
        return None


def resize_video(input_file_name):
    """
    Function to resize the input video to 864x480 resolution.

    Args:
        input_file_name (str): Name of the input file.
    """
    if input_file_name is None:
        return

    try:
        input_path = os.path.join(INPUT_FOLDER, input_file_name)
        video = cv2.VideoCapture(input_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        resized_video_path = os.path.join(RESIZED_FOLDER, input_file_name)
        out = cv2.VideoWriter(resized_video_path, fourcc, fps, (864, 480))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(frame, (864, 480), interpolation=cv2.INTER_CUBIC)
            out.write(frame)

        video.release()
        out.release()
    except Exception as e:
        print(f"Error in resizing video: {e}")


def generate_frames():
    """
    Function to generate individual frames from the resized video.
    """
    input_file_name = get_input_file_name()
    if input_file_name is None:
        return

    try:
        os.makedirs(FRAMES_FOLDER, exist_ok=True)
        input_path = os.path.join(RESIZED_FOLDER, input_file_name)
        video = cv2.VideoCapture(input_path)
        count = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame_path = os.path.join(FRAMES_FOLDER, f"{count}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
        video.release()
    except Exception as e:
        print(f"Error in generating frames: {e}")


def combine_predicted_outputs(input_file_name):
    """
    Function to combine frames with predicted outputs into a final video.

    Args:
        input_file_name (str): Name of the input file.
    """
    if input_file_name is None:
        return

    try:

        os.makedirs(RESIZED_FOLDER, exist_ok=True)
        # frame_path = os.path.join(PREDICTED_OUTPUT_FOLDER, input_file_name)
        frame_path = PREDICTED_OUTPUT_FOLDER
        frames = [os.path.join(frame_path, frame) for frame in sorted(os.listdir(frame_path), key=lambda x: int(x.split(".")[0]))]

        first_frame = cv2.imread(frames[0])
        height, width, layers = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        output_video_path = os.path.join(FINAL_VIDEO_FOLDER, input_file_name)
        video = cv2.VideoWriter(output_video_path, fourcc, 25.0, (width, height))

        for frame in frames:
            video.write(cv2.imread(frame))

        cv2.destroyAllWindows()
        video.release()
    except Exception as e:
        print(f"Error in combining predicted outputs: {e}")
