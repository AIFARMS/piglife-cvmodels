import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import os
import time

from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', 
    '--input', 
    default='input/inference_data/video_1.mp4', 
    help='path to the input data'
)
parser.add_argument(
    '-t', 
    '--threshold', 
    default=0.5, 
    type=float,
    help='score threshold for discarding detection'
)
parser.add_argument(
    '-w',
    '--weights',
    default='out/checkpoint.pth',
    help='path to the trained wieght file'
)
parser.add_argument(
    '--show',
    action='store_true',
    help='whether to visualize the results in real-time on screen'
)
args = parser.parse_args()

OUT_DIR = os.path.join('outputs', 'inference')
os.makedirs(OUT_DIR, exist_ok=True)

model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    pretrained=False, num_classes=91
)

# model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(class_names), bias=True)
# model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(class_names)*4, bias=True)
# model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(class_names), kernel_size=(1, 1), stride=(1, 1))
print(model)

# initialize the model
ckpt = torch.load(args.weights)
model.load_state_dict(ckpt['model'])
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

cap = cv2.VideoCapture(args.input)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))
save_name = args.input.split(os.path.sep)[-1].split('.')[0]
# Define codec and create VideoWriter object.
out = cv2.VideoWriter(
    f"{OUT_DIR}/{save_name}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, 
    (frame_width, frame_height)
)

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        image = frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        # keep a copy of the original image for OpenCV functions and applying masks
        orig_image = image.copy()
        # transform the image
        image = transform(image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        start_time = time.time()
        masks, boxes, labels = get_outputs(image, model, args.threshold)
        end_time = time.time()
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        print(f"Frame {frame_count}, FPS: {fps:.1f}")
        result = draw_segmentation_map(orig_image, masks, boxes, labels)
        cv2.putText(
            result,
            text=f"{fps:.1f} FPS",
            org=(15, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        out.write(result)
        # visualize the image
        if args.show:
            cv2.imshow('Result', np.array(result))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

# Release VideoCapture().
cap.release()
# Close all frames and video windows.
cv2.destroyAllWindows()

# Calculate and print the average FPS.
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")