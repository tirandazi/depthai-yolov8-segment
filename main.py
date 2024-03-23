import json

import cv2
import depthai as dai
import numpy as np

from yolo_api import Segment

with open("helpers/config.json", "r") as config:
    model_data = json.load(config)

preview_img_width = model_data["input_width"]
preview_img_height = model_data["input_height"]
input_shape = [1, 3, preview_img_height, preview_img_width]

output0_shape = model_data["shapes"]["output0"]
output1_shape = model_data["shapes"]["output1"]

path_to_yolo_blob = "models/yolov8n-seg.blob"


def main():
    pipeline = dai.Pipeline()

    # Init pipeline's output queue
    xout_rgb = pipeline.createXLinkOut()
    xout_yolo_nn = pipeline.createXLinkOut()
    xout_depth = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    xout_yolo_nn.setStreamName("yolo_nn")
    xout_depth.setStreamName("depth")

    # Neural network pipeline properties
    nn = pipeline.createNeuralNetwork()
    nn.setBlobPath(path_to_yolo_blob)
    nn.out.link(xout_yolo_nn.input)

    # Color cam properties (Cam_A/RGB)
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(preview_img_width, preview_img_height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
    cam_rgb.preview.link(nn.input)
    cam_rgb.preview.link(xout_rgb.input)
    print("Color cam resolution: ", cam_rgb.getResolutionSize())

    # Stereo/Depth node properties
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(preview_img_width, preview_img_height)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.depth.link(xout_depth.input)

    # Left cam properties (Cam_B/Mono)
    left = pipeline.create(dai.node.MonoCamera)
    left.setCamera("left")
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    left.out.link(stereo.left)
    print("Left cam resolution: ", left.getResolutionSize())

    # Right cam properties (Cam_C/Mono)
    right = pipeline.create(dai.node.MonoCamera)
    right.setCamera("right")
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
    right.out.link(stereo.right)
    print("Right cam resolution: ", right.getResolutionSize())

    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        yolo_nn_queue = device.getOutputQueue("yolo_nn", maxSize=4, blocking=False)
        depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)

        while True:
            rgb_queue_msg = rgb_queue.get()
            yolo_nn_queue_msg = yolo_nn_queue.get()
            depth_queue_msg = depth_queue.get()

            if rgb_queue_msg is not None:
                frame = rgb_queue_msg.getCvFrame()
                depth_frame = (
                    depth_queue_msg.getFrame() if depth_queue_msg is not None else None
                )  # depth frame returns values in mm (millimeter)

                if yolo_nn_queue_msg is not None and depth_frame is not None:
                    output0 = np.reshape(
                        yolo_nn_queue_msg.getLayerFp16("output0"),
                        newshape=(output0_shape),
                    )
                    output1 = np.reshape(
                        yolo_nn_queue_msg.getLayerFp16("output1"),
                        newshape=(output1_shape),
                    )
                    if len(output0) > 0 and len(output1) > 0:
                        yoloseg = Segment(
                            input_shape=input_shape,
                            input_height=preview_img_height,
                            input_width=preview_img_width,
                            conf_thres=0.2,
                            iou_thres=0.5,
                        )
                        yoloseg.prepare_input_for_oakd(frame.shape[:2])
                        boxes, scores, class_ids, mask_maps = (
                            yoloseg.segment_objects_from_oakd(output0, output1)
                        )
                        # frame = yoloseg.draw_masks(frame.copy())
                        for i in range(len(boxes)):
                            mask = mask_maps[i].astype(np.uint8)
                            contours, _ = cv2.findContours(
                                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            for contour in contours:
                                center, size, angle = cv2.minAreaRect(contour)
                                box_points = cv2.boxPoints((center, size, angle))
                                box_points = np.int32(box_points)
                                cv2.drawContours(frame, [box_points], 0, (0, 255, 0), 2)

                                # print("RotatedRect: ", center, size, angle)
                                # print("box points: ", box_points)

                                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                                cv2.drawContours(mask, [box_points], -1, 255, -1)

                                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                                cv2.imshow("Masked Output", masked_frame)

                                masked_depth_frame = cv2.bitwise_and(
                                    depth_frame, depth_frame, mask=mask
                                )
                                vals_inside_rotated_rect = masked_depth_frame[mask != 0]
                                depth = np.median(vals_inside_rotated_rect)

                                # print(vals_inside_rotated_rect)
                                # print(len(vals_inside_rotated_rect))

                                # depth_frame_roi = depth_frame[
                                #     int(boxes[i][1]) : int(boxes[i][3]),
                                #     int(boxes[i][0]) : int(boxes[i][2]),
                                # ]
                                # depth = np.median(depth_frame_roi.flatten())
                                # print(len(depth_frame_roi.flatten()))
                                # print(len(depth_frame.flatten()))

                                cv2.putText(
                                    frame,
                                    f"Z: {int(depth)} mm",
                                    (int(center[0]) - 50, int(center[1]) - 25),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (255, 255, 255),
                                )
                                cv2.putText(
                                    frame,
                                    f"Angle: {round(angle, 1)}",
                                    (int(center[0]) - 50, int(center[1]) + 25),
                                    cv2.FONT_HERSHEY_TRIPLEX,
                                    0.5,
                                    (255, 255, 255),
                                )
                cv2.imshow("Output", frame)

            else:
                print("rgb_queue message is empty")

            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    main()
