from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    model = YOLO('yolov8n.pt')

    model.train(
        data=r'C:\Users\Public\Documents\Project_detect_emotion\data_detect.yaml',
        epochs=50,
        imgsz=640,
        project=r'C:\Users\Public\Documents\Project_detect_emotion',
        name='face_dect_train'
    )

if __name__ == '__main__':
    main()
