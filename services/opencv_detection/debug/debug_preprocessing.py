from pathlib import Path
import cv2

from services.opencv_detection.preprocessing import ImagePreprocessor

IMAGE_PATH = Path("/home/sharan/Documents/python/incense_stick_quality_control/dataset/test/good/20260728_064107.jpg")

# IMAGE_PATH = Path("/home/sharan/Documents/python/incense_stick_quality_control/dataset/test/defective/IMG-20260728-WA0002.jpg")

OUTPUT_DIR  = Path("/home/sharan/Documents/python/incense_stick_quality_control/runs/opencv_detection/preprocessing/good")

def main() -> None:
    """Main function to run the image preprocessing."""
    preprocessor = ImagePreprocessor()
    preprocessed_images = preprocessor.preprocess(IMAGE_PATH)

    # Save the preprocessed images
    for name, image in preprocessed_images.items():
        output_path = OUTPUT_DIR / f"{name}.png"
        cv2.imwrite(str(output_path), image)
        print(f"Saved {name} to {output_path}")

if __name__ == "__main__":
    main()
