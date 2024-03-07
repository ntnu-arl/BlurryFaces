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

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

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
    dy = min(int(0.2 * (y2 - y1)), 20)
    dx = min(int(0.2 * (x2 - x1)), 20)
    # crop the image due to the current box
    y1 = max(0, y1 - dy)
    y2 = min(y2 + dy, image.shape[0]-1)
    x1 = max(0, x1 - dx)
    x2 = min(x2 + dx, image.shape[1]-1)
    sub = image[y1:y2, x1:x2]

    # apply GaussianBlur on cropped area
    blur = cv2.blur(sub, (25, 25))

    # paste blurred image on the original image
    image[y1:y2, x1:x2] = blur

    # cv2.rectangle(image, (y1, x1), (y2, x2), (0,0,0), -1)

  return image

def getMaskImage(image, boxes):
  masked_image = np.zeros((image.shape[0], image.shape[1],3), np.uint8)
  for box in boxes:
    # unpack each box
    x1, y1 = box["x1"], box["y1"]
    x2, y2 = box["x2"], box["y2"]
    dy = min(int(0.2 * (y2 - y1)), 20)
    dx = min(int(0.2 * (x2 - x1)), 20)
    # crop the image due to the current box
    y1 = max(0, y1 - dy)
    y2 = min(y2 + dy, image.shape[0]-1)
    x1 = max(0, x1 - dx)
    x2 = min(x2 + dx, image.shape[1]-1)
    cv2.rectangle(masked_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
  
  return masked_image
# config_file_name = "/home/mihir/post_proc_ws/src/bag_utils/bag_processor/configs/bwt_dataset.yaml"

bag_file_name = "/home/nkhedekar/workspaces/BlurryFaces/bags.csv"
img_topics = ["/blackfly_left/blackfly_left", "/blackfly_right/blackfly_right", "/zed_stereo/left/image_raw", "/zed_stereo/right/image_raw"]

bag_files = []
with open(bag_file_name, "r") as fd:
  bag_files = fd.read().splitlines()


model_path = "/home/nkhedekar/workspaces/BlurryFaces/face_model/face.pb"
threshold = 0.2

# create detection object
detector = Detector(model_path=model_path, name="detection")

for bag in bag_files:
  print("opening bag: ", bag)
  in_bag = rosbag.Bag(bag)
  print("Now processing: ", bag)
  filename, file_extension = os.path.splitext(bag)
  out_bag_name = filename + "_face_blurred_with_mask" + file_extension
  out_bag = rosbag.Bag(out_bag_name, 'w')
  for topic, msg, t in tqdm(in_bag.read_messages(), total=in_bag.get_message_count()):
    if topic in img_topics:
      cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
      grayscale = False
      if (len(cv_image.shape) == 2):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        grayscale = True
      
      faces = detector.detect_objects(cv_image, threshold=threshold)
      # apply blurring
      blur_image = cv_image.copy()
      blur_image = blurBoxes(blur_image, faces)
      mask_image = getMaskImage(cv_image, faces)

      if grayscale:
        # Convert the image back to grayscale
        blur_image = cv2.cvtColor(blur_image, cv2.COLOR_RGB2GRAY)

      img_msg = bridge.cv2_to_imgmsg(blur_image)
      img_msg.header = msg.header
      if grayscale:
        img_msg.encoding = "mono8"
      else:
        img_msg.encoding = "rgb8"

      mask_img_msg = bridge.cv2_to_imgmsg(mask_image)
      mask_img_msg.header = msg.header
      mask_img_msg.encoding = "mono8"
      out_bag.write(topic, img_msg, t)
      out_bag.write(topic + "_mask", mask_img_msg, t)
    # else:
    #   out_bag.write(topic, msg, t)
    # print(LINE_UP, end=LINE_CLEAR)
  out_bag.close()

  cv2.destroyAllWindows()


# change to non compressed image
# 