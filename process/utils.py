from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from pathlib import Path


def update_traffic_light(frame, traffic_light_status):
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)

    color = green_color if traffic_light_status == 'Green' else red_color

    cv2.putText(frame, traffic_light_status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return 1 if traffic_light_status == 'Green' else 0



model = YOLO('../model/yolov8n.pt')

def count_cars_in_frame(frame):

    
    
    classNames = ["person","bicycle","car","motorcycle","airplane","bus","train","truck",
              "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
              "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
              "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
              "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
              "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
              "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
              "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
              "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
              "teddy bear","hair drier","toothbrush"]

    car_count = 0  # Biến để đếm số lượng xe
    is_train_detected = False

    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1
            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            currentClass = classNames[cls]
            # if currentClass == 'car' and conf > 0.3:
            #     car_count += 1  # Tăng số lượng xe lên 1
            #     cv2.putText(frame, f'Car Count: {car_count}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if (currentClass == 'car' or currentClass == 'truck' or currentClass == 'Motorcycle' or currentClass == 'Bus') and conf > 0.1:
                car_count += 1
                cvzone.cornerRect(frame, (x1, y1, w, h))
            elif currentClass == 'train' and conf > 0.1:
                is_train_detected = True
                cvzone.cornerRect(frame, (x1, y1, w, h))
    print(is_train_detected)
    return car_count,is_train_detected






def create_quad_display(video_paths):
    cv2.namedWindow('Quad Display', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Quad Display', 960, 720)

    video_captures = [cv2.VideoCapture(video_path) for video_path in video_paths]

    while True:
        frames = []

        for cap in video_captures:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break
            frames.append(frame)

        if len(frames) != 4:
            break

        h, w, _ = frames[0].shape
        h, w = h // 2, w // 2
        frames_resized = [cv2.resize(frame, (w, h)) for frame in frames]

        quad_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        car_counts = []
        is_train_detected_list = []
        
        for i, frame in enumerate(frames_resized):
            car_count, is_train_detected = count_cars_in_frame(frame)
            car_counts.append(car_count)
            is_train_detected_list.append(is_train_detected)

            x, y = i % 2, i // 2
            y1, y2 = y * h, (y + 1) * h
            x1, x2 = x * w, (x + 1) * w
            quad_frame[y1:y2, x1:x2] = frame

        # Xác định trạng thái đèn giao thông dựa trên quy tắc mới
        if any(is_train_detected_list):
            # Trường hợp có tàu hỏa, ảnh có tàu hỏa sẽ có đèn xanh, các ảnh khác có đèn đỏ
            for i, frame in enumerate(frames_resized):
                traffic_light_status = 'Green' if is_train_detected_list[i] else 'Red'
                update_traffic_light(frame, traffic_light_status)
                cv2.putText(quad_frame, f'Traffic Light {i + 1}: {traffic_light_status}', (20, 20 + (i + 1) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Trường hợp không có tàu hỏa, ảnh có số lượng xe lớn nhất sẽ có đèn xanh, các ảnh khác có đèn đỏ
            max_car_count_index = np.argmax(car_counts)
            for i, frame in enumerate(frames_resized):
                traffic_light_status = 'Green' if i == max_car_count_index else 'Red'
                update_traffic_light(frame, traffic_light_status)
                cv2.putText(quad_frame, f'Traffic Light {i + 1}: {traffic_light_status}', (20, 20 + (i + 1) * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for i, frame in enumerate(frames_resized):
            x, y = i % 2, i // 2
            y1, y2 = y * h, (y + 1) * h
            x1, x2 = x * w, (x + 1) * w
            quad_frame[y1:y2, x1:x2] = frame

        cv2.imshow('Quad Display', quad_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    for cap in video_captures:
        cap.release()
    cv2.destroyAllWindows()






# def create_quad_display(video_paths):
#     # Tạo cửa sổ
#     cv2.namedWindow('Quad Display', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Quad Display', 960, 720)  # Adjust the window size

#     # Tạo các đối tượng VideoCapture cho 4 video
#     video_captures = [cv2.VideoCapture(video_path) for video_path in video_paths]

#     while True:
#         frames = []

#         for cap in video_captures:
#             ret, frame = cap.read()
#             if not ret or frame is None or frame.size == 0:
#                 break
#             frames.append(frame)

#         # Check if any frames are empty or have an invalid size
#         if len(frames) != 4:
#             break

#         # Resize frames to match the quadrant size
#         h, w, _ = frames[0].shape
#         h, w = h // 2, w // 2
#         frames_resized = [cv2.resize(frame, (w, h)) for frame in frames]

#         # Create an empty frame with four quadrants
#         quad_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

#         car_counts = []
#         for i, frame in enumerate(frames_resized):
#             car_count = count_cars_in_frame(frame)
#             car_counts.append(car_count)

#             x, y = i % 2, i // 2
#             y1, y2 = y * h, (y + 1) * h
#             x1, x2 = x * w, (x + 1) * w
#             quad_frame[y1:y2, x1:x2] = frame

#         # Display car counts on the frame
#         for i, count in enumerate(car_counts):
#             cv2.putText(quad_frame, f'Car Count {i + 1}: {count}', (20, 20 + i * 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#         cv2.imshow('Quad Display', quad_frame)

#         phim_bam = cv2.waitKey(1)
#         if phim_bam == ord('q'):
#             break

#     # Đóng tất cả video capture
#     for cap in video_captures:
#         cap.release()
#     cv2.destroyAllWindows()


