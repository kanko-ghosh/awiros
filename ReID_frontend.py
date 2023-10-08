from ultralytics import YOLO
import cv2
import math 
import numpy as np
# start webcam
import time
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person"]


def get_id_of_person(frame, no_of_persons, raw_images):
    """gets the person id from the ReID microservice

    Args:
        frame (int): the frame number sent to the microservice
        no_of_persons (int): the number of persons in the frame
        raw_images (list<ndarray>): a list of numpy array representing the images of the persons

    Returns:
        list<str, float>: returns a list of person ids and the confidence of the prediction
    """

    res = []
    for e in range(no_of_persons):
        res.append(["test"+str(e), np.random.rand()])
    return res



while True:
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)

    # coordinates
    count = 0
    ls = []
    dim = []
    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            if cls > 0:
                continue
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

            # put box in cam
            temp = img[y1:y2, x1:x2]
            if temp.shape[0] == 0 or temp.shape[1] == 0:
                continue
            ls.append(temp)
            dim.append((x1, y1))
    
    result = get_id_of_person(count, len(ls), ls)

    for id, d in zip(result, dim):
        org = [d[0], d[1]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        cv2.putText(img, "ID: "+id[0]+"->"+str(id[1])[:3], org, font, fontScale, color, thickness)

    try:
        cv2.imshow('Webcam', img)
    except:
        print(img.shape)
        exit()
    if cv2.waitKey(1) == ord('q'):
        break

    count += 1

cap.release()
cv2.destroyAllWindows()