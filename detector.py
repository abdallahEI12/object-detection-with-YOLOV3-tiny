import cv2
import numpy as np
import os

class Detector:
    def __init__(self):
        #initialize the model
        self.net = cv2.dnn.readNet(r'yolov3.weights', r'yolov3.cfg')


        #get all the classes from coco names
        self.classes = []
        with open(r'tiny yolo\coco.names', 'r') as file:
            self.classes = [line.strip() for line in file.readlines()]
        print(self.classes)

        self.layer_names = self.net.getUnconnectedOutLayersNames()
        self.confusion_matrix = {
            "tp":0,
            'fp':0,
            'tn':0,
            'fn':0
        }


    def get_blobs_and_classes(self,folder_path):
        images_and_classes = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
                image_path = os.path.join(folder_path, file_name)
                image = cv2.imread(image_path)

                image_class_file = os.path.join(folder_path, os.path.splitext(file_name)[0] + '.txt')
                with open(image_class_file,'r') as file:
                    image_class = [int(line.split(' ')[0]) for line in file.readlines()]
                images_and_classes.append((image,image_class))

        return images_and_classes

    def detect(self,images_and_classes):
        for image , c in images_and_classes:
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            height = image.shape[0]
            width = image.shape[1]
            results = self.net.forward(self.layer_names)
            class_ids = []
            confidences = []
            boxes = []
            for result in results:
                for detection in result:

                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            human_counts = 0
            others_count = 0
            if len(indices) == 0:
                self.confusion_matrix['fn'] += len(c)
            else:
                for index in indices:
                    predicted_class = self.classes[class_ids[index]]
                    if predicted_class == 'person':
                        human_counts+=1
                    else:
                        others_count +=1
            self.confusion_matrix['tn'] += others_count
            self.confusion_matrix['tp'] += human_counts
            if len(c) > human_counts:
                self.confusion_matrix['fn'] += len(c)-human_counts
            for i in indices:
                 box = boxes[i]
                 label = self.classes[class_ids[i]]
                 confidence = confidences[i]
                 cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                 cv2.putText(image, f"{label} {confidence:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLO Object Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def evaluate(self):
        print(self.confusion_matrix)





