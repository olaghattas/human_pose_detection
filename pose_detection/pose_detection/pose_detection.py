import threading

import cv2
import mediapipe as mp
import rclpy
from cv_bridge import CvBridge
# from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image
import math
from detection_msgs.msg import PoseMsg
# from rclpy.exceptions import ParameterNotDeclaredException
from typing import Union
from std_msgs.msg import Int32MultiArray

class PoseDetection(Node):
    def __init__(self):
        super().__init__('detect_pose')

        self.pub_ = self.create_publisher(PoseMsg, '/coord_shoulder_joint_in_px', 10)

        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.camera_callback_dining,
            10)
        self.subscription  # prevent unused variable warning
        self.camera_name = "name"
        self.br = CvBridge()

        self.mpPose = mp.solutions.pose
        # self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

        self.pose = self.mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.image = None
        self.plotting = True

    def camera_callback_dining(self, data):

        frame = self.br.imgmsg_to_cv2(data)

        # Setup mediapipe instance

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = self.pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Extract landmarks
        # print('pose', results.pose_landmarks)

        self.image = image
        if results.pose_landmarks is not None:

            landmarks = results.pose_landmarks.landmark

            # Render detections

            self.mpDraw.draw_landmarks(self.image, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                       self.mpDraw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       self.mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            _PRESENCE_THRESHOLD = 0.5
            _VISIBILITY_THRESHOLD = 0.5
            landmark_list = results.pose_landmarks
            image_rows, image_cols, _ = self.image.shape
            idx_to_coordinates = {}
            for idx, landmark in enumerate(landmark_list.landmark):
                if ((landmark.HasField('visibility') and
                     landmark.visibility < _VISIBILITY_THRESHOLD) or
                        (landmark.HasField('presence') and
                         landmark.presence < _PRESENCE_THRESHOLD)):
                    continue
                # ld_px_x, ld_px_y = self._normalized_to_pixel_coordinates(
                #     landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,
                #     landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value].y, image_cols, image_rows)

            ld_px_x, ld_px_y = _normalized_to_pixel_coordinates(
                landmarks[self.mpPose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[self.mpPose.PoseLandmark.RIGHT_KNEE.value].y, image_cols, image_rows)
            print(ld_px_x, ld_px_y)

            # Render the specific landmark points
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2)

            msg = PoseMsg()
            msg.name = self.camera_name

            point_msg = Int32MultiArray()
            point_msg.data = [ld_px_x, ld_px_y]

            msg.pixel_coordinates = point_msg
            # Publish the message
            print('################# Knee #############', point_msg)
            self.pub_.publish(msg)
            # Setup status box


def plot(pose_detection):
    while pose_detection.plotting:
        if pose_detection.image is not None:
            # cv2.rectangle(pose_detection.image, (0, 0), (225, 73), (245, 117, 16), -1)

            cv2.imshow('Mediapipe Feed', pose_detection.image)
            cv2.waitKey(100)


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def main(args=None):
    rclpy.init(args=None)

    pose_detection = PoseDetection()

    t1 = threading.Thread(target=plot, args=(pose_detection,))
    t1.start()
    print('mains')
    rclpy.spin(pose_detection)
    pose_detection.plotting = False
    t1.join()


if __name__ == '__main__':
    main()
