import cv2
import fastseg as fs
from torchvision.transforms import ToTensor, Resize
from PIL import Image
import torch
import numpy as np
from skimage.morphology import medial_axis
from skimage import img_as_bool
from scipy.ndimage import distance_transform_edt


num_classes = 35
kernel_size = 5
line_width = int(512 * 0.02)
line_color = (0, 0, 255)

class_names = ['unlabeled', 'flat-road', 'flat-sidewalk', 'flat-crosswalk', 'flat-cyclinglane', 'flat-parkingdriveway', 'flat-railtrack', 'flat-curb', 'human-person', 'human-rider', 'vehicle-car', 'vehicle-truck', 'vehicle-bus', 'vehicle-tramtrain', 'vehicle-motorcycle', 'vehicle-bicycle', 'vehicle-caravan', 'vehicle-cartrailer', 'construction-building', 'construction-door', 'construction-wall', 'construction-fenceguardrail', 'construction-bridge', 'construction-tunnel', 'construction-stairs', 'object-pole', 'object-trafficsign', 'object-trafficlight', 'nature-vegetation', 'nature-terrain', 'sky', 'void-ground', 'void-dynamic', 'void-static', 'void-unclear']


def load_model():
    device = torch.device("cpu")
    model = fs.MobileV3Large(num_classes=num_classes)
    model.load_state_dict(torch.load("../Model/fast_scnn_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(frame):
    image = Image.fromarray(frame)
    image = image.convert('RGB')
    img_transform = Resize((512, 512))
    to_tensor_transform = ToTensor()
    image = img_transform(image)
    image_tensor = to_tensor_transform(image).unsqueeze(0)
    return image_tensor


def get_sidewalk_mask(label_map):
    sidewalk_index = class_names.index('flat-sidewalk')
    sidewalk_mask = label_map == sidewalk_index
    sidewalk_matrix = np.zeros_like(sidewalk_mask, dtype=np.uint8)
    sidewalk_matrix[sidewalk_mask] = 255
    return sidewalk_matrix


def process_sidewalk_matrix(sidewalk_matrix, width, height):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    sidewalk_matrix = cv2.resize(sidewalk_matrix, (width, height))
    sidewalk_matrix = cv2.erode(sidewalk_matrix, kernel)
    sidewalk_matrix = cv2.dilate(sidewalk_matrix, kernel)
    return sidewalk_matrix


def find_sidewalk_contour(sidewalk_matrix):
    contours, _ = cv2.findContours(sidewalk_matrix, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None


def get_skeleton(contour_mask):
    contour_mask_bool = img_as_bool(contour_mask)
    skeleton = medial_axis(contour_mask_bool)
    return skeleton


def filter_skeleton(skeleton, threshold=70):
    distances = distance_transform_edt(skeleton)
    filtered_skeleton = skeleton & (distances > np.percentile(distances, threshold))
    return filtered_skeleton

def main(video_file):
    model = load_model()

    cap = cv2.VideoCapture(video_file)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_tensor = preprocess_image(frame)

        with torch.no_grad():
            output = model(image_tensor)

        label_map = output.argmax(dim=1).squeeze().cpu().numpy()
        sidewalk_matrix = get_sidewalk_mask(label_map)
        sidewalk_matrix = process_sidewalk_matrix(sidewalk_matrix, width, height)
        largest_contour = find_sidewalk_contour(sidewalk_matrix)

        if largest_contour is not None:
            contour_mask = np.zeros_like(sidewalk_matrix)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

            skeleton = get_skeleton(contour_mask)
            filtered_skeleton = filter_skeleton(skeleton)

            filtered_skeleton_uint8 = (filtered_skeleton.astype(np.uint8) * 255)
            skeleton_contours, _ = cv2.findContours(filtered_skeleton_uint8, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)

            if skeleton_contours:
                longest_contour = max(skeleton_contours, key=lambda x: cv2.arcLength(x, True))
                cv2.drawContours(frame, [longest_contour], -1, line_color, line_width)

            # concatenate original frame and sidewalk mask horizontally
            mask_frame = cv2.cvtColor(sidewalk_matrix, cv2.COLOR_GRAY2BGR)
            concat_frame = np.concatenate((frame, mask_frame), axis=1)
            cv2.imshow('Masked Image', concat_frame)
        else:
            cv2.imshow('Masked Image', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = "../Data/Konstanzenstrasse-Nürnberg SAMSUNG-S7.mp4"
    main(video_file)
