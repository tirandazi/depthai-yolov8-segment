# YOLO Segment & Depth | OAK-D Pro PoE

**Notice:** As of March 26, 2024, the depthai-sdk does not offer official support for running YOLO instance segmentation inference alongside depth on the device.

This repository provides code that facilitates inference on YOLOv8 models, extracting segmentation masks and depth information in millimeters (mm). The depth is extracted as the median of depth points within the rotated rectangle region of segment masks.

## Running Inference

Follow these steps to execute the inference on a custom-trained YOLOv8 model:

1. Open `helpers/export.py`.
2. Modify the dimensions (height & width) values as required.
3. Run the script. This action will generate a file in ONNX format.
4. Note down the values of shapes for input, output0 and output1 from the terminal output. Open `helpers/config.json` and update the corresponding values. Also, update the class names in order in the `class_names` array.
5. Next, Open a web browser and navigate to [http://blobconverter.luxonis.com/](http://blobconverter.luxonis.com/).
6. Select the OpenVino default version.
7. Choose ONNX as the model source and click continue.
8. Upload the ONNX file.
9. In the model optimizer params, update it to `--data_type=FP16 --mean_values=[0,0,0] --scale_values=[255,255,255]` and click convert.
10. This will download the `.blob` file.
11. Move/Copy this blob file to the `models` directory.
12. Update the blob path in the variable `path_to_yolo_blob` at line 19 in `main.py`.

## Credits

Special thanks to pedro-UCA and jakaskerl for the support provided in the Luxonis forum.

---
**Note:** For any further assistance, please refer to the Luxonis forum.
