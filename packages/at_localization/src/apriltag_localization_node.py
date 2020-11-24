#!/usr/bin/env python3
import numpy as np
import cv2
import rospy
from duckietown.dtros import DTROS, NodeType
from geometry_msgs.msg import TransformStamped, Quaternion, Vector3, Transform
from tf.transformations import quaternion_from_matrix, quaternion_from_euler, rotation_matrix
import tf
import tf2_ros
from dt_apriltags import Detector
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

from geometry import SE3, rotation_translation_from_SE3, SE3_from_rotation_translation, rotation_from_axis_angle


class ApriltagLocalizationNode(DTROS):

    def __init__(self, node_name):

        super(ApriltagLocalizationNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh_name = rospy.get_namespace().strip("/")
        self.radius = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius', 100)
        self.baseline = 0.0968
        self.rectify_flag = rospy.get_param(f'/{self.veh_name}/{node_name}/rectify', 100)
        self.bridge = CvBridge()
        self.image = None
        self.image_timestamp = rospy.Time.now()
        self.cam_info = None
        self.camera_info_received = False
        self.newCameraMatrix = None
        self.at_detector = Detector(families='tag36h11',
                                    nthreads=1,
                                    quad_decimate=1.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)

        trans_base_cam = np.array([0.0582, 0.0, 0.1072])
        rot_base_cam = rotation_from_axis_angle(np.array([0, 1, 0]), np.radians(15))
        rot_cam_base = rot_base_cam.T
        trans_cam_base = - rot_cam_base @ trans_base_cam
        self.pose_cam_base = SE3_from_rotation_translation(rot_cam_base, trans_cam_base)
        self.tfs_msg_cam_base = TransformStamped()
        self.tfs_msg_cam_base.header.frame_id = 'camera'
        self.tfs_msg_cam_base.header.stamp = rospy.Time.now()
        self.tfs_msg_cam_base.child_frame_id = 'at_baselink'
        self.tfs_msg_cam_base.transform = self.pose2transform(self.pose_cam_base)
        self.static_tf_br_cam_base = tf2_ros.StaticTransformBroadcaster()

        translation_map_at = np.array([0.0, 0.0, 0.08])
        rot_map_at = np.eye(3)
        self.pose_map_at = SE3_from_rotation_translation(rot_map_at, translation_map_at)
        self.tfs_msg_map_april = TransformStamped()
        self.tfs_msg_map_april.header.frame_id = 'map'
        self.tfs_msg_map_april.header.stamp = rospy.Time.now()
        self.tfs_msg_map_april.child_frame_id = 'apriltag'
        self.tfs_msg_map_april.transform = self.pose2transform(self.pose_map_at)
        self.static_tf_br_map_april = tf2_ros.StaticTransformBroadcaster()

        self.tfs_msg_april_cam = TransformStamped()
        self.tfs_msg_april_cam.header.frame_id = 'apriltag'
        self.tfs_msg_april_cam.header.stamp = rospy.Time.now()
        self.tfs_msg_april_cam.child_frame_id = 'camera'
        self.br_april_cam = tf.TransformBroadcaster()

        self.tfs_msg_map_base = TransformStamped()
        self.tfs_msg_map_base.header.frame_id = 'map'
        self.tfs_msg_map_base.header.stamp = rospy.Time.now()
        self.tfs_msg_map_base.child_frame_id = 'at_baselink'

        R_x_c = rotation_from_axis_angle(np.array([1, 0, 0]), np.pi / 2)
        R_z_c = rotation_from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        R_y_a = rotation_from_axis_angle(np.array([0, 1, 0]), -np.pi / 2)
        R_z_a = rotation_from_axis_angle(np.array([0, 0, 1]), np.pi / 2)
        R_dtc_c = R_x_c @ R_z_c
        R_a_dta = R_y_a @ R_z_a
        self.T_dtc_c = SE3_from_rotation_translation(R_dtc_c, np.array([0, 0, 0]))
        self.T_a_dta = SE3_from_rotation_translation(R_a_dta, np.array([0, 0, 0]))


        self.sub_image_comp = rospy.Subscriber(
            f'/{self.veh_name}/camera_node/image/compressed',
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1
        )

        self.sub_cam_info = rospy.Subscriber(f'/{self.veh_name}/camera_node/camera_info', CameraInfo,
                                             self.cb_camera_info, queue_size=1)

        self.pub_tf_at_loc = rospy.Publisher(f'/{self.veh_name}/{node_name}/transform_stamped',
                                             TransformStamped, queue_size=10)

    def image_cb(self, image_msg):
        try:
            image = self.readImage(image_msg)
        except ValueError as e:
            self.logerr('Could not decode image: %s' % e)
            return

        self.image = image
        self.image_timestamp = image_msg.header.stamp

    def cb_camera_info(self, msg):
        if not self.camera_info_received:
            self.cam_info = msg

        self.camera_info_received = True

    def detect_april_tag(self):
        if self.rectify_flag:
            K = self.newCameraMatrix
        else:
            K = np.array(self.cam_info.K).reshape((3, 3))
        camera_params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        img_grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(img_grey, True, camera_params, 0.065)
        list_pose_tag = []
        for tag in tags:
            pose_SE3 = SE3_from_rotation_translation(tag.pose_R, np.squeeze(tag.pose_t))
            list_pose_tag.append(pose_SE3)

        return list_pose_tag

    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []

    def pose2transform(self, pose: SE3) -> Transform:
        rot, trans = rotation_translation_from_SE3(pose)
        quat_tf = quaternion_from_matrix(pose)
        quaternion = Quaternion(quat_tf[0], quat_tf[1], quat_tf[2], quat_tf[3])
        translation = Vector3(trans[0], trans[1], trans[2])
        return Transform(translation, quaternion)

    def pose_inverse(self, pose: SE3) -> Transform:
        rot, trans = rotation_translation_from_SE3(pose)
        rot_inv = rot.T
        trans_inv = -rot_inv @ trans
        return SE3_from_rotation_translation(rot_inv, trans_inv)

    def rectify_image(self):
        cameraMatrix = np.array(self.cam_info.K).reshape((3, 3))
        distCoeffs = np.array(self.cam_info.D)
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix,
                                                        distCoeffs,
                                                        (640, 480),
                                                        1.0)
        self.newCameraMatrix = newCameraMatrix
        mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix,
                                    distCoeffs,
                                    np.eye(3),
                                    newCameraMatrix,
                                    (640, 480),
                                    cv2.CV_32FC1)
        self.image = cv2.remap(self.image, mapx, mapy, cv2.INTER_LINEAR)

    def onShutdown(self):
        super(ApriltagLocalizationNode, self).onShutdown()

    def run(self):
        while not rospy.is_shutdown():
            if self.image is None:
                continue
            if not self.camera_info_received:
                continue

            if self.rectify_flag:
                self.rectify_image()
            list_pose_tag = self.detect_april_tag()
            if not list_pose_tag:
                continue

            pose_dta_dtc = self.pose_inverse(list_pose_tag[0])
            pose_april_cam = self.T_a_dta @ pose_dta_dtc @ self.T_dtc_c
            pose_map_base = self.pose_map_at @ pose_april_cam @ self.pose_cam_base
            self.tfs_msg_map_base.transform = self.pose2transform(pose_map_base)
            self.tfs_msg_map_base.header.stamp = self.image_timestamp
            self.pub_tf_at_loc.publish(self.tfs_msg_map_base)

            self.tfs_msg_map_april.header.stamp = self.image_timestamp
            self.static_tf_br_map_april.sendTransform(self.tfs_msg_map_april)

            self.tfs_msg_cam_base.header.stamp = self.image_timestamp
            self.static_tf_br_cam_base.sendTransform(self.tfs_msg_cam_base)

            self.tfs_msg_april_cam.transform = self.pose2transform(pose_april_cam)
            self.tfs_msg_april_cam.header.stamp = self.image_timestamp
            self.br_april_cam.sendTransformMessage(self.tfs_msg_april_cam)




if __name__ == '__main__':
    # Initialize the node
    apriltag_node = ApriltagLocalizationNode(node_name='encoder_localization_node')
    apriltag_node.run()
    rospy.on_shutdown(apriltag_node.onShutdown)
    # Keep it spinning to keep the node alive
    rospy.spin()
