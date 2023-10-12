findString = "/content/PyTorch-Simple-MaskRCNN/data/coco2017/train2017"
replaceString = "/Users/jenish/desktop/20230807_Fine_Tune_Mask_RCNN_PyTorch_on_Custom_Dataset/input/pig-segmentation/train2017"
  
with open('pig_coco_train.json', 'r') as f:
    data = f.read()
    data = data.replace(findString, replaceString)
    # print(data.count("/content/PyTorch-Simple-MaskRCNN/data/coco2017/train2017"))
  
with open('pig_coco_train.json', 'w') as f:
    f.write(data)
    
print("JSON Data replaced")
