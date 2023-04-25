import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, ToTensor
import fastseg as fs
import math

num_classes = 35
kernel_size = 5
line_width = int(512 * 0.02)
line_color = (0, 0, 255)

class_names = ['unlabeled', 'flat-road', 'flat-sidewalk', 'flat-crosswalk', 'flat-cyclinglane', 'flat-parkingdriveway', 'flat-railtrack', 'flat-curb', 'human-person', 'human-rider', 'vehicle-car', 'vehicle-truck', 'vehicle-bus', 'vehicle-tramtrain', 'vehicle-motorcycle', 'vehicle-bicycle', 'vehicle-caravan', 'vehicle-cartrailer', 'construction-building', 'construction-door', 'construction-wall', 'construction-fenceguardrail', 'construction-bridge', 'construction-tunnel', 'construction-stairs', 'object-pole', 'object-trafficsign', 'object-trafficlight', 'nature-vegetation', 'nature-terrain', 'sky', 'void-ground', 'void-dynamic', 'void-static', 'void-unclear']


def load_model(model_path):
    device = torch.device("cpu")
    model = fs.MobileV3Large(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
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


def find_mid_point(largest_contour, height, tolerance=15, min_y_distance=30):
    left_points = {}
    right_points = {}

    for point in largest_contour:
        x, y = point[0]
        if y not in left_points or x < left_points[y]:
            left_points[y] = x
        if y not in right_points or x > right_points[y]:
            right_points[y] = x

    middle_points = []
    for y in range(height - 1, -1, -1):
        if y in left_points and y in right_points:
            middle_x = (left_points[y] + right_points[y]) // 2
            if y < height - min_y_distance and abs(middle_x - left_points[y]) > tolerance and abs(middle_x - right_points[y]) > tolerance:
                middle_points.append((middle_x, y))

    return middle_points



def draw_middle_dots(frame, middle_points):
    for i in range(len(middle_points) - 1):
        cv2.line(frame, middle_points[i], middle_points[i + 1], (0, 0, 255), 2)


def angle_between_points(p1, p2, p3):
    v1 = [p2[0] - p1[0], p2[1] - p1[1]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
    mag_v2 = math.sqrt(v2[0] * v2[0] + v2[1] * v2[1])

    angle = math.acos(dot_product / (mag_v1 * mag_v2))
    angle = math.degrees(angle)

    return angle



def main(model_path, video_file):
    model = load_model(model_path)
    if video_file == "0":
        video_file = int(video_file)
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    # Get the frame rate and size of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/Users/yatin/Documents/SENSATION/SENSATION-1.1/Data/output_video.mp4', fourcc, fps,
                          (width, height))

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
        min_y_distance = 100
        if largest_contour is not None:
            # print(largest_contour)
            frame = cv2.resize(frame, (width, height))
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 255), 2)
            middle_points = find_mid_point(largest_contour, height)
            if len(middle_points) > 1:
                # Find the middle point of the contour after min_y_distance pixels
                first_middle_point_after_min_y = None
                for point in middle_points:
                    if height - point[1] >= min_y_distance:
                        first_middle_point_after_min_y = point
                        break

                if first_middle_point_after_min_y is not None:
                    frame_bottom_middle_point = (width // 2, height - 1)
                    delta_x = first_middle_point_after_min_y[0] - frame_bottom_middle_point[0]
                    direction = "right" if delta_x > 0 else "left"
                    delta_x = abs(delta_x)

                    angle_to_turn = math.degrees(math.atan2(delta_x, min_y_distance))
                    if angle_to_turn > 5:
                        print(
                            "Please turn the camera {} by {:.2f} degrees to align with the middle of the detected line.".format(
                                direction, angle_to_turn))

                    # Draw a line connecting the bottom middle point of the frame and the first middle point of the detected line after min_y_distance pixels
                    cv2.line(frame, frame_bottom_middle_point, first_middle_point_after_min_y, (0, 255, 0), 2)

            draw_middle_dots(frame, middle_points)
            cv2.imshow('Sidewalk Contours', frame)
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # count += 1
        # if count == 15:
        #     break

    cap.release()
    cv2.destroyAllWindows()

import argparse

parser = argparse.ArgumentParser(description='Process video file with sidewalk detection model.')
parser.add_argument('model_path', type=str, help='Path to the sidewalk detection model')
parser.add_argument('video_file', type=str, help='Path to the input video file')

args = parser.parse_args()
# if __name__ == "__main__":
#     video_file = "../Data/Konstanzenstrasse-Nürnberg-SAMSUNG-S7.mp4"
#     main(video_file)
if __name__ == "__main__":
    main(args.model_path, args.video_file)
