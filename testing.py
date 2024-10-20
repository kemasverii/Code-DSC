# Testing hasil training model dari file train_data.ipynb

from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak bisa membuka kamera.")
    exit()
cap.set(3, 720)
cap.set(4, 720)
cap.set(5, 720)

# model
model1= YOLO('D:/Coding/Python/src/DSC/runs/detect/train11/weights/last.pt')
# object classes
classNames = ["100rb","10rb","1rb","20rb","2rb","50rb","5rb"]

def run_model(no_model):
    while True:
        success, img = cap.read()
        results = no_model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
run_model(model1)

