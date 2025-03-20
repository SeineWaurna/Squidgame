from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov11n.pt')
    results = model.train(data='datasets/data.yaml', epochs=100, imgsz=640)