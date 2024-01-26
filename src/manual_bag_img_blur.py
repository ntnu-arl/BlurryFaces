#!/usr/bin/env python3

import sys
import yaml
import rosbag
import numpy as np
from tqdm import tqdm
import os
import cv2
from DetectorAPI import Detector
from cv_bridge import CvBridge
bridge = CvBridge()

# def blurBoxes(image, boxes):
#     """
#     Argument:
#     image -- the image that will be edited as a matrix
#     boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys

#     Returns:
#     image -- the blurred image as a matrix
#     """

#     for box in boxes:
#         # unpack each box
#         x1, y1 = box["x1"], box["y1"]
#         x2, y2 = box["x2"], box["y2"]

#         # crop the image due to the current box
#         sub = image[y1:y2, x1:x2]

#         # apply GaussianBlur on cropped area
#         blur = cv2.blur(sub, (25, 25))

#         # paste blurred image on the original image
#         image[y1:y2, x1:x2] = blur

#         # cv2.rectangle(image, (y1, x1), (y2, x2), (0,0,0), -1)

#     return image

def blurBoxes(image, boxes):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred, each box must be int the format (x_top_left, y_top_left, width, height)

    Returns:
    image -- the blurred image as a matrix
    """

    for box in boxes:
        # unpack each box
        x, y, w, h = [d for d in box]

        # crop the image due to the current box
        sub = image[y:y+h, x:x+w]

        # apply GaussianBlur on cropped area
        blur = cv2.GaussianBlur(sub, (23, 23), 30)

        # paste blurred image on the original image
        image[y:y+h, x:x+w] = blur

    return image

bag_file_name = "/home/mihir/source_code/BlurryFaces/bags_manual.csv"
img_topic = "/blackfly_image"

bag_files = []
with open(bag_file_name, "r") as fd:
  bag_files = fd.read().splitlines()

print("Loaded bag files")
# create detection object
# detector = Detector(model_path=model_path, name="detection")

for bag in bag_files:
  in_bag = rosbag.Bag(bag)
  print("Now processing: ", bag)
  filename, file_extension = os.path.splitext(bag)
  out_bag_name = filename + "_proc" + file_extension
  out_bag = rosbag.Bag(out_bag_name, 'w')
  print("Created out bag")
  for topic, msg, t in tqdm(in_bag.read_messages(), total=in_bag.get_message_count()):
    if topic == img_topic:
      cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
      ROIs = []
      temp_image = cv_image.copy()
      # keep getting ROIs until pressing 'q'
      while True:
        # get ROI cv2.selectROI(window_name, image_matrix, selecting_start_point)
        box = cv2.selectROI('blur', temp_image, fromCenter=False)
        empty = True
        for e in box:
          if e != 0:
            empty = False
        if empty:
          break
        # add selected box to box list
        ROIs.append(box)
        # draw a rectangle on selected ROI
        cv2.rectangle(temp_image, (box[0], box[1]),
                      (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 3)
        print('ROI is saved, press q to stop capturing, press any other key to select other ROI')
        # if 'q' is pressed then break
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break
      
      print("ROIs: ", ROIs)
      # apply blurring
      blur_image = cv_image.copy()
      if not len(ROIs) == 0:
        blur_image = blurBoxes(blur_image, ROIs)
        # cv2.imshow('blurred', blur_image)
        # cv2.waitKey(0)

      img_msg = bridge.cv2_to_imgmsg(blur_image, encoding="passthrough")
      img_msg.header = msg.header
      out_bag.write("/blackfly_image", img_msg, t)
    else:
      out_bag.write(topic, msg, t)
  out_bag.close()

  cv2.destroyAllWindows()