import cv2
import numpy as np
import torch

from class_names import INSTANCE_CATEGORY_NAMES as coco_names

np.random.seed(0)

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the model.
        outputs = model(image)
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    
    labels = [coco_names[i] for i in outputs[0]['labels']]
    for i in range(len(labels)):
        if i!=0 or i!=1:
            labels[i] == 0
    # labels = [0 for i in labels if (i!=0 or i!=1)]
    print(outputs[0]['labels'])
    return masks, boxes, labels

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1.0
    beta = 0.5 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    #convert the original PIL image into NumPy format
    image = np.array(image)
    # convert from RGN to OpenCV BGR format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i in range(len(masks)):
        # apply a randon color mask to each object
        color = COLORS[coco_names.index(labels[i])]
        if masks[i].any() == True:
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            # apply mask on the image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

            # draw the bounding boxes around the objects
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                        thickness=2)
            # put the label text above the objects
            cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                        thickness=2, lineType=cv2.LINE_AA)
    return image