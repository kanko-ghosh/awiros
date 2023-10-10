from typing import Any
from ultralytics import YOLO
import cv2
import math 
import numpy as np
import time
import tensorflow as tf
import PIL
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# fps = cap.get(cv2.CAP_PROP_FPS)

def load_mob_classif_torch_model(mob_classif_model_path):
    # Load the trained model
    print("Loading mob model ...")
    model_val = models.resnet18(pretrained=False)
    model_val.fc = nn.Linear(model_val.fc.in_features, 2)
    model_val.load_state_dict(torch.load(mob_classif_model_path, map_location=torch.device('cpu')))
    model_val.eval()
    return model_val


class HelmetIO:
    def __init__(self,yolo_model_path, full_body_yolo_model_path, classif_model_path,mob_classif_model_path):
        self.yolo_model = YOLO(yolo_model_path)
        self.full_body_yolo_model = YOLO(full_body_yolo_model_path)
        self.classif_model = tf.keras.models.load_model(classif_model_path)
        self.mob_classif_model = load_mob_classif_torch_model(mob_classif_model_path)
        print("Models loaded")
        self.classNames = ["person"]
        self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x / 255.0),  # Scale pixel values to [0, 1]
            ])

    def process_yolo_resut(self, results,img):
        ls = []
        dim = []
        b_count = 0
        for r in results:
            boxes = r.boxes

            for box in boxes:
                b_count += 1
                print("box")
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

        if b_count == 0:
            print("No person found")
        return ls, dim

    

    def classify_person_list(self, person_ls):
        res = []
        for person in person_ls:
            person = cv2.resize(person, (64, 64))
            person = np.reshape(person, (1, 64, 64, 3))
            person = person / 255.0
            pred = self.classif_model(person, training=False)
            res.append("Wearing Helmet" if pred[0][0] < 0.3 else "Not Wearing Helmet ... Chacha Bidhayak Hai")
        return res

    def classify_mobile_usage(self, person_ls):
        res = []
        for person in person_ls:
            person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
            person = PIL.Image.fromarray(person).convert("RGB")

            with torch.no_grad():   
                person = self.transform(person)
                person = person.unsqueeze(0)
                output = self.mob_classif_model(person)

            _,predicted_class = torch.max(output.data, 1)
            pred = predicted_class.item()
            res.append("No Mobile usage" if pred == 0 else "Using Mobile ... Chacha Bidhayak Hai")
        return res
    

    def annotate_frame(self, frame, dim, classif_result, font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 0, 0),thickness = 2):
        for classif_res, d in zip(classif_result, dim):
            org = [d[0], d[1]]
            cv2.putText(frame, classif_res, org, font, fontScale, color, thickness)
        return frame

    def write_frames_to_video(self,frames, output_video_path, frame_rate=30.0, output_resolution=None):
        if len(frames) == 0:
            print("Error: Empty frames list.")
            return

        # Get the height and width from the first frame
        frame_height, frame_width, _ = frames[0].shape

        # If output resolution is specified, use it; otherwise, use the frame's resolution
        if output_resolution is not None:
            frame_width, frame_height = output_resolution

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

        for frame in frames:
            if frame.shape[:2] != (frame_height, frame_width):
                # Resize the frame if it doesn't match the desired resolution
                frame = cv2.resize(frame, (frame_width, frame_height))

            # Write the frame to the video file
            out.write(frame)

    def convert_v_to_v(self, src_path, dest_path, read_fps = 10.0):
        cap = cv2.VideoCapture(src_path)
        cap.set(3, 640)
        cap.set(4, 480)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame2s = []
        # Calculate the frame skip interval to achieve the target FPS
        frame_skip_interval = int(fps / read_fps)
        print("Frame skip interval: " + str(frame_skip_interval))
        # Iterating through the frames
        frame_count = 0
        while True:
            try:
                frame_count += 1
                # Skip frames to achieve target FPS
                if frame_count % frame_skip_interval != 0:
                    continue
                print("Reading frame ... " + str(frame_count))
                success, img = cap.read()
                if not success:
                    print("Not success")
                    break
                print("Running yolo ...")
                results = self.yolo_model(img, stream=True, verbose=False)
                full_body_results = self.full_body_yolo_model(img, stream=True, verbose=False)

                # For facial recognition for helmet usage
                print("Processing yolo result ...")
                person_ls, dim = self.process_yolo_resut(results,img)
                print("Classifying person list ...")
                classif_result = self.classify_person_list(person_ls)
                print("Annotating frame ...")
                frame1 = self.annotate_frame(img, dim, classif_result)

                # For full body recognition for mobile usage
                print("Processing full body yolo result ...")
                person_ls, dim = self.process_yolo_resut(full_body_results,img)
                print("Classifying person list ...")
                classif_result = self.classify_mobile_usage(person_ls)
                print("Annotating frame ...")
                frame2 = self.annotate_frame(frame1, dim, classif_result,color = (0, 255, 0),thickness = 3)

                if src_path == 0:
                    cv2.imshow('Webcam', frame2)
                    cv2.waitKey(1)
                    
                frame2s.append(frame2)

            except Exception as e:
                print(e)
                break
        print("Done reading frames.")
        cap.release()
        frame2s = np.array(frame2s)

        # Writing the video
        print("Writing video in original fps ...")
        self.write_frames_to_video(frame2s, dest_path, fps, (640, 480))
        return dest_path
    

if __name__ == "__main__":
    print("Starting...")
    hi = HelmetIO(yolo_model_path="./models/debas_is_the_goat_model.pt",full_body_yolo_model_path ="./models/yolov8m.pt", classif_model_path="./models/hel0_model.h5", mob_classif_model_path="models/model16.pt")
    hi.convert_v_to_v(src_path=0, dest_path="./data/output.mp4", read_fps=1.0)
    print("Kabhi Alvida Na Kehna")