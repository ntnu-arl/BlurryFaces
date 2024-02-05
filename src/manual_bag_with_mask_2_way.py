import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import copy
import os

key_for = 'd'
key_back = 'a'
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'
MAX_BUFFER = 20

class RosbagPlayer:
    def __init__(self, bag_file):
        self.bag_file = bag_file
        self.bag = rosbag.Bag(bag_file)
        self.bridge = CvBridge()
        self.image_iterator = iter(self.get_messages("/blackfly_image/compressed"))
        self.image_mask_iterator = iter(self.get_messages("/blackfly_image_mask/compressed"))
        # self.image_iterator = iter(self.get_messages("/blackfly_image"))
        # self.image_mask_iterator = iter(self.get_messages("/blackfly_image_mask"))
        self.image_buffer = []
        self.mask_buffer = []
        self.output_image_buffer = []
        self.output_mask_buffer = []
        self.max_buffer_length = MAX_BUFFER
        self.absolute_image_counter = 0
        self.absolute_iterator_counter = -1
        self.buffer_index_counter = 0
        self.iterator_reached_end = False

    def get_messages(self, topic):
        for _, msg, t in self.bag.read_messages(topics=[topic]):
            yield (msg,t)
    
    def blurBoxes(self, image, boxes):
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
            blur = cv2.GaussianBlur(sub, (25, 25), 30)

            # paste blurred image on the original image
            image[y:y+h, x:x+w] = blur

        return image

    def process(self):
        filename, file_extension = os.path.splitext(self.bag_file)
        out_bag_name = filename + "_verified_images_only" + file_extension
        out_bag = rosbag.Bag(out_bag_name, 'w')
        while True:
            print(LINE_UP, end=LINE_CLEAR)
            print("img:", self.absolute_image_counter, "| iter:", self.absolute_iterator_counter, "| buffer:", self.buffer_index_counter)
            if self.absolute_iterator_counter < self.absolute_image_counter:
                try:
                    image = next(self.image_iterator)
                    mask_image = next(self.image_mask_iterator)
                    self.image_buffer.append(image)
                    self.mask_buffer.append(mask_image)
                    image_copy = copy.deepcopy(image)
                    mask_copy = copy.deepcopy(mask_image)
                    self.output_image_buffer.append(image_copy)
                    self.output_mask_buffer.append(mask_copy)
                    # out_image = self.bridge.cv2_to_compressed_imgmsg(self.bridge.imgmsg_to_cv2(image_copy[0], desired_encoding="passthrough"))
                    # out_mask = self.bridge.cv2_to_compressed_imgmsg(self.bridge.imgmsg_to_cv2(mask_copy[0], desired_encoding="passthrough"))
                    # self.output_image_buffer.append((out_image, image_copy[1]))
                    # self.output_mask_buffer.append((out_mask, mask_copy[1]))
                    if len(self.image_buffer) > self.max_buffer_length:
                        self.image_buffer.pop(0)
                    if len(self.mask_buffer) > self.max_buffer_length:
                        self.mask_buffer.pop(0)
                    if len(self.output_image_buffer) > self.max_buffer_length:
                        out = self.output_image_buffer.pop(0)
                        out_bag.write("/blackfly_image/compressed", out[0], out[1])
                    if len(self.output_mask_buffer) > self.max_buffer_length:
                        out = self.output_mask_buffer.pop(0)
                        out_bag.write("/blackfly_image_mask/compressed", out[0], out[1])
                    self.absolute_iterator_counter += 1
                except StopIteration:
                    self.iterator_reached_end = True
            
            current_image = self.image_buffer[self.buffer_index_counter]
            current_image_cv = self.bridge.compressed_imgmsg_to_cv2(current_image[0], desired_encoding="passthrough")
            current_mask = self.mask_buffer[self.buffer_index_counter]
            current_mask_cv = self.bridge.compressed_imgmsg_to_cv2(current_mask[0], desired_encoding="passthrough")
            # current_image = self.image_buffer[self.buffer_index_counter]
            # current_image_cv = self.bridge.imgmsg_to_cv2(current_image[0], desired_encoding="passthrough")
            # current_mask = self.mask_buffer[self.buffer_index_counter]
            # current_mask_cv = self.bridge.imgmsg_to_cv2(current_mask[0], desired_encoding="passthrough")

            output_image = self.bridge.compressed_imgmsg_to_cv2(self.output_image_buffer[self.buffer_index_counter][0], desired_encoding="passthrough").copy()
            output_mask = self.bridge.compressed_imgmsg_to_cv2(self.output_mask_buffer[self.buffer_index_counter][0], desired_encoding="passthrough").copy()

            # cv2.imshow("Image", current_image_cv)
            cv2.imshow("2 Output Image", output_image)
            cv2.imshow("2 Output Mask", output_mask)
            key = cv2.waitKey(0)
            # key = 83
            ROIs = []
            temp_image = current_image_cv.copy()
            # keep getting ROIs until pressing 'q'
            if key & 0xFF == ord('f'):
                while True:
                    # get ROI cv2.selectROI(window_name, image_matrix, selecting_start_point)
                    box = cv2.selectROI("2 Output Image", temp_image, fromCenter=False)
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
                    print('ROI is saved, press b/right arrow/left arrowd to stop capturing, press any other key to select other ROI')
                    # if 'q' is pressed then break
                    key = cv2.waitKey(0)
                    if key & 0xFF == ord('b') or key & 0xFF == ord(key_for) or key & 0xFF == ord(key_back) or key == 27:
                        break
                    
                print("ROIs: ", ROIs)

                # apply blurring
                output_mask = current_mask_cv.copy()
                output_image = current_image_cv.copy()
                if not len(ROIs) == 0:
                    output_image = self.blurBoxes(output_image, ROIs)
                    for box in ROIs:
                        cv2.rectangle(output_mask, (box[0], box[1]),(box[0]+box[2], box[1]+box[3]), (255, 255, 255), -1)

            output_image_msg = self.bridge.cv2_to_compressed_imgmsg(output_image)
            output_image_msg.header = current_image[0].header
            output_mask_msg = self.bridge.cv2_to_compressed_imgmsg(output_mask)
            output_mask_msg.header = current_image[0].header
            self.output_image_buffer[self.buffer_index_counter] = (output_image_msg, self.output_image_buffer[self.buffer_index_counter][1])
            self.output_mask_buffer[self.buffer_index_counter] = (output_mask_msg, self.output_image_buffer[self.buffer_index_counter][1])

            if key == 27:  # Press ESC to exit
                break
            elif key & 0xFF == ord(key_for): 
                self.absolute_image_counter += 1
                self.buffer_index_counter += 1
                if self.buffer_index_counter >= self.max_buffer_length:
                    self.buffer_index_counter = self.max_buffer_length - 1
                if self.buffer_index_counter == self.max_buffer_length - 1 and self.iterator_reached_end:
                    break
            elif key & 0xFF == ord(key_back): 
                self.buffer_index_counter -= 1
                if self.buffer_index_counter < 0:
                    self.buffer_index_counter = 0
                else:
                    self.absolute_image_counter -= 1
        
        for i in range(0, len(self.output_image_buffer)):
            out_bag.write("/blackfly_image/compressed", self.output_image_buffer[i][0], self.output_image_buffer[i][1])
            out_bag.write("/blackfly_image_mask/compressed", self.output_mask_buffer[i][0], self.output_mask_buffer[i][1])
        
        out_bag.close()
            
            
    
    def play(self):
        for image_msg, image_mask_msg in zip(self.image_iterator, self.image_mask_iterator):
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
            image_mask = self.bridge.imgmsg_to_cv2(image_mask_msg, desired_encoding="passthrough")

            # Display images or do any processing here
            cv2.imshow("Image", image)
            cv2.imshow("Image Mask", image_mask)

            key = cv2.waitKey(0)

            if key == 27:  # Press ESC to exit
                break
            elif key == 81:  # Press left arrow key
                try:
                    next(self.image_iterator)
                    next(self.image_mask_iterator)
                except StopIteration:
                    break
            elif key == 83:  # Press right arrow key
                pass  # Continue with the next images

        cv2.destroyAllWindows()
        self.bag.close()

if __name__ == "__main__":
    bag_file = "/media/mihir/bb6291c1-0351-4346-9db7-611dd5f66757/home/arl/ROSBAGS/bwt_dataset/PK/3_comp_bottom_bilge_section/processed/a_2024-01-22-20-29-53_face_blurred_with_mask_verified_images_only.bag"
    player = RosbagPlayer(bag_file)
    player.process()
    # player.play()