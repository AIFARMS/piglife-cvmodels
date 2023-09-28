import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import glob
import os

from PIL import Image
from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
from class_names import INSTANCE_CATEGORY_NAMES as class_names

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', 
    '--input', 
    required=True, 
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

# initialize the model
ckpt = torch.load(args.weights)
model.load_state_dict(ckpt['model'])
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()
print(model)

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

image_paths = glob.glob(os.path.join(args.input, '*.jpg'))
for image_path in image_paths:
    print(image_path)
    image = Image.open(image_path)
    # keep a copy of the original image for OpenCV functions and applying masks
    orig_image = image.copy()
    
    # transform the image
    image = transform(image)
    # add a batch dimension
    image = image.unsqueeze(0).to(device)
    
    masks, boxes, labels = get_outputs(image, model, args.threshold)
    
    result = draw_segmentation_map(orig_image, masks, boxes, labels)
    
    # visualize the image
    if args.show:
        cv2.imshow('Segmented image', np.array(result))
        cv2.waitKey(0)
    
    # set the save path
    save_path = f"{OUT_DIR}/{image_path.split(os.path.sep)[-1].split('.')[0]}.jpg"
    cv2.imwrite(save_path, result)