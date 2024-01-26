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

def blurBoxes(image, boxes):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys

    Returns:
    image -- the blurred image as a matrix
    """

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]

        # apply GaussianBlur on cropped area
        blur = cv2.blur(sub, (25, 25))

        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur

        # cv2.rectangle(image, (y1, x1), (y2, x2), (0,0,0), -1)

    return image


# config_file_name = "/home/mihir/post_proc_ws/src/bag_utils/bag_processor/configs/bwt_dataset.yaml"

bag_file_name = "/home/mihir/source_code/BlurryFaces/bags_mjolnir.csv"
img_topics = ["/blackfly_left/blackfly_left", "/blackfly_left/blackfly_right"]

topics = ["/blackfly_left/camera_info",
          "/blackfly_right/camera_info",
          "/compslam/odometry",
          "/laser_mapping_path",
          "/os_cloud_node/points",
          "/radar/left/cloud",
          "/radar/right/cloud",
          "/tf",
          "/tf_static",
          "/vectornav_node/imu",
          "/vectornav_node/uncomp_imu"]
# Create a dictionary of topic_group_name : list of topics in that group
# separations = {}

# with open(config_file_name) as f:
#   docs = list(yaml.load_all(f, Loader=yaml.FullLoader))
#   for k, v in docs[0].items():
#     separations[k] = v

bag_files = []
with open(bag_file_name, "r") as fd:
  bag_files = fd.read().splitlines()


model_path = "/home/mihir/source_code/BlurryFaces/face_model/face.pb"
threshold = 0.2

# create detection object
detector = Detector(model_path=model_path, name="detection")

for bag in bag_files:
  in_bag = rosbag.Bag(bag)
  print("Now processing: ", bag)
  filename, file_extension = os.path.splitext(bag)
  out_bag_name = filename + "_proc" + file_extension
  out_bag = rosbag.Bag(out_bag_name, 'w')
  for topic, msg, t in tqdm(in_bag.read_messages(), total=in_bag.get_message_count()):
    if topic in img_topics:
      cv_image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
      faces = detector.detect_objects(cv_image, threshold=threshold)
      # apply blurring
      blur_image = blurBoxes(cv_image, faces)
      # cv2.imshow('blured', blur_image)
      # key = cv2.waitKey(1)
      img_msg = bridge.cv2_to_imgmsg(blur_image, encoding="passthrough")
      img_msg.header = msg.header
      out_bag.write("/blackfly_image", img_msg, t)
    elif topic in topics:
      out_bag.write(topic, msg, t)
    # else:
    #   out_bag.write(topic, msg, t)
  out_bag.close()

  cv2.destroyAllWindows()