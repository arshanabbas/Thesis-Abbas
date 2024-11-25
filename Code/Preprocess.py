import cv2
import matplotlib.pyplot as plt
import glob
import os
import json
from pycocotools.coco import COCO
import json

img_files = glob.glob(r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\COCO1\images\*.jpg")
print(img_files)

num_elements = len(img_files)
print(num_elements)

for img_file in img_files:
    img = cv2.imread(img_file)
    center = img.shape
    h = img.shape[1]
    y = center[0]/2 - h/2

    crop_img = img[int(y):int(y+h),:]

    print(img_file.replace("images","cropped_images"))

print(cv2.imwrite(os.path.join(r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\COCO1\images_cropped",img_file.split("\\")[-1]), crop_img))

cv2.imwrite("Test.jpg", crop_img)

img_file

img_file.replace("images","cropped_images") #DON'T RUN THIS COMMAND

with open(r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\COCO1\Test\result.json") as file:
    data = json.load(file)

data.keys()

len(data['categories'])

coco=COCO(r"F:\Pomodoro\Work\TIME\Script\Thesis-Abbas-Segmentation\COCO1\Test\resultcopy.json")

for key, value in coco.imgs.items():
    tmp_width = value['width']
    value['width'] = value['height']
    value['height'] = tmp_width-(882*2)

for key, value in coco.anns.items():
    x_val = value['segmentation'][0][0::2]
    y_val = value['segmentation'][0][1::2]
    new_pairs = [None]*(len(x_val)+len(y_val))
    new_pairs[::2] = [i * (2268/4032) for i in x_val]
    new_pairs[1::2] = [(i * (4032/2268))-882 for i in y_val]
    value['segmentation'][0] = new_pairs
    
    tmp_bbox = value['bbox']
    value['bbox'] = [tmp_bbox[0]*(2268/4032),(tmp_bbox[1]*(4032/2268))-882,tmp_bbox[2]*(2268/4032),tmp_bbox[3]*(4032/2268)]

coco_dict = coco.dataset

# Write the dictionary to the output file
with open('new_result.json', 'w') as f:
    json.dump(coco_dict, f)

