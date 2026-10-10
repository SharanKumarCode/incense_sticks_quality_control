from services.opencv_detection.detector import OpenCVDetector

def main():
    detector = OpenCVDetector()
    result = detector.detect("/home/sharan/Documents/python/incense_stick_quality_control/input/damaged/IMG-20260728-WA0000.jpg")

    if result.is_defective:
        print(f"Defects: {result.detected_defects}")
        print(f"Measurements: {result.measurements}")

if  __name__ == "__main__":
    main()