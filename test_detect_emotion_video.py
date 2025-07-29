import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from torchvision import models
from torch import nn

# ---------- Thiết bị ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Danh sách nhãn ----------
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
gender_classes = ['Female', 'Male']

# ---------- Load YOLOv8 để detect khuôn mặt ----------
yolo_model = YOLO(r"C:\Users\Public\Documents\Project_detect_emotion\face_dect_train\weights\best.pt")

# ---------- Load mô hình phân loại cảm xúc ----------
emotion_model = models.resnet50(pretrained=False)
emotion_model.fc = nn.Linear(2048, len(emotion_classes))
checkpoint_emotion = torch.load(r"C:\Users\Public\Documents\Project_detect_emotion\trained_models_emotion\best.pt", map_location=device)
emotion_model.load_state_dict(checkpoint_emotion["model_state_dict"])
emotion_model.to(device)
emotion_model.eval()

# ---------- Load mô hình phân loại giới tính ----------
gender_model = models.resnet50(pretrained=False)
gender_model.fc = nn.Linear(gender_model.fc.in_features, len(gender_classes))
checkpoint_gender = torch.load(r"C:\Users\Public\Documents\Project_detect_emotion\trained_models_gender\best.pt", map_location=device)
gender_model.load_state_dict(checkpoint_gender["model_state_dict"])
gender_model.to(device)
gender_model.eval()

# ---------- Transform cho khuôn mặt ----------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
# ---------- Đường dẫn video input & output ----------
video_path = r"C:\Users\Public\Documents\Project_detect_emotion\test\test_video\video1.mp4"
output_video_path = r"C:\Users\Public\Documents\Project_detect_emotion\test\output\result_video.mp4"

# ---------- Khởi tạo VideoCapture ----------
cap = cv2.VideoCapture(video_path)

# ---------- Khởi tạo VideoWriter ----------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Hết video

    # Detect khuôn mặt
    results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        input_tensor = transform(face_pil).unsqueeze(0).to(device)

        # --------- Dự đoán cảm xúc ---------
        with torch.no_grad():
            output_emotion = emotion_model(input_tensor)
            pred_emotion = torch.argmax(output_emotion, dim=1).item()
            emotion = emotion_classes[pred_emotion]

        # --------- Dự đoán giới tính ---------
        with torch.no_grad():
            output_gender = gender_model(input_tensor)
            pred_gender = torch.argmax(output_gender, dim=1).item()
            gender = gender_classes[pred_gender]

        # --------- Vẽ kết quả ---------
        label_text = f"{gender}, {emotion}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Ghi frame ra file
    out.write(frame)

    # Hiển thị nếu muốn
    cv2.imshow("Detected", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
