from ultralytics import YOLO

def main():
    MODEL_PATH = "/home/sharan/Documents/python/incense_stick_quality_control/runs/yolo-detection/baseline/weights/best.pt"
    model = YOLO(MODEL_PATH)
    print(model.names)


if __name__  == "__main__":
    main()
