import os
import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Open video file
video_path = os.getcwd() + "/gym_video.mp4"
cap = cv2.VideoCapture(video_path)

detected_people = []  # 감지된 사람 정보를 저장할 리스트
min_duration = 3.0  # 최소 머무르는 시간 설정 (3초 이상)
last_detected_time = 0  # 마지막으로 인식된 시간을 저장할 변수

frame_num = 0  # 프레임 번호 초기화

MachineA = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1  # 프레임 번호 증가

    height, width = frame.shape[0], frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0은 person 클래스
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

    detected_people_in_frame = []  # 현재 프레임에서 감지된 사람 정보를 저장할 리스트

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)
            detected_people_in_frame.append((frame_num, x, y, w, h))  # 프레임 번호와 사람 위치 추가

    if detected_people_in_frame:
        current_time = time.time()
        if current_time - last_detected_time >= min_duration:
            detected_people.append(detected_people_in_frame)
            last_detected_time = current_time

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 감지된 사람 정보 출력
for people in detected_people:
    for frame_num, x, y, w, h in people:
        if 200 < x < 400 and 400 < y < 500:
            MachineA += 1
            print("==================================================")
            print(f"Frame {frame_num} - Person detected at ({x}, {y}) for {min_duration:.2f} seconds - MachineA Num = {MachineA}")
            MachineA = 0

"""
import os
import cv2
import img as img
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Open video file
video_path = os.getcwd() + "/gym_video.mp4"
cap = cv2.VideoCapture(video_path)

detected_people = []  # 감지된 사람 정보를 저장할 리스트
min_duration = 3  # 최소 머무르는 시간 (단위: 초)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[0], frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0은 person 클래스
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

    detected_people_in_frame = []  # 현재 프레임에서 감지된 사람 정보를 저장할 리스트

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)
            detected_people_in_frame.append((label, confidences[i], x, y, w, h))
            # print(label, confidences[i], x, y, w, h)
            # print("Person Detected at (x={}, y={}, w={}, h={})".format(x, y, w, h))

    if detected_people_in_frame:
        detected_people.append(detected_people_in_frame)

    frame_num = 1
    # 감지된 사람 정보 출력
    for people in detected_people:
        print(f"Frame {frame_num} - Detected People:")
        frame_num += 1
        for person in people:
            label, confidence, x, y, w, h = person
            print(f"Label: {label}, Confidence: {confidence:.2f}, x: {x}, y: {y}, w: {w}, h: {h}")

            # 특정 좌표에 대해 머무른 시간 계산
            if 200< x <400 and 400 <y <500:
                current_time = time.time()
                if "last_detected_time" not in locals():
                    last_detected_time = current_time
                else:
                    duration = current_time - last_detected_time
                    if duration >= min_duration:
                        print("==================================================")
                        print(f"Person detected at ({x}, {y}) for {duration:.2f} seconds")
                        print("==================================================\n")
                        last_detected_time = current_time
                        break

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""

"""
import os
import cv2
import img as img
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Open video file
video_path = os.getcwd() + "/gym_video.mp4"
cap = cv2.VideoCapture(video_path)

detected_people = []  # 감지된 사람 정보를 저장할 리스트

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[0], frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0은 person 클래스
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

    detected_people_in_frame = []  # 현재 프레임에서 감지된 사람 정보를 저장할 리스트

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)
            detected_people_in_frame.append((label, confidences[i], x, y, w, h))
            # print(label, confidences[i], x, y, w, h)
            # print("Person Detected at (x={}, y={}, w={}, h={})".format(x, y, w, h))

    if detected_people_in_frame:
        detected_people.append(detected_people_in_frame)


    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 감지된 사람 정보 출력
for frame_num, people in enumerate(detected_people, start=1):
    print(f"Frame {frame_num} - Detected People:")
    for person in people:
        label, confidence, x, y, w, h = person
        print(f"Label: {label}, Confidence: {confidence:.2f}, x: {x}, y: {y}, w: {w}, h: {h}")
"""

"""
# 이미지 경로 설정
image_path = os.getcwd() + "/gym.jpg"
# 이미지 로드
image = cv2.imread(image_path)
# 이미지 크기 가져오기
height, width = image.shape[0], image.shape[1]

# 이미지 전처리 및 모델에 입력
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# 감지된 객체 정보 저장
class_ids = []
confidences = []
boxes = []

# 객체 감지
for out in outs:
    for detection in out:
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

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), font, 1, color, 2)
        print(label,confidences[i],x,y,w,h)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
