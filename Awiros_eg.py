print("lol")

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(
        ['BikesHelmets0.png', 'BikesHelmets0.png'],
        stream=True
    )  # return a generator of Results objects

# Process results generator
for result in results:
    print(result)
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs