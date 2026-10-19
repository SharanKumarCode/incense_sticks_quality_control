from pathlib import Path
import cv2

from services.opencv_detection.preprocessing import ImagePreprocessor
from services.opencv_detection.segmentation import find_stick_candidates, draw_candidates, select_primary_stick, create_stick_mask

IMAGE_PATH = Path("/home/sharan/Documents/python/incense_stick_quality_control/dataset/test/good/20260728_064117.jpg")

IMAGE_PATH = Path("/home/sharan/Documents/python/incense_stick_quality_control/dataset/train/defective/IMG-20260728-WA0006.jpg")

def main() -> None:
    """Main function to run the image preprocessing and contour detection."""
    preprocessor = ImagePreprocessor()
    results = preprocessor.preprocess(IMAGE_PATH)

    masks = results.get("adaptive_inverted")
    if masks is not None:
        candidates = find_stick_candidates(masks)
        primary_stick = select_primary_stick(candidates)
        visualization = draw_candidates(results["resized"], candidates)

        cv2.imwrite(
                    str(Path(f"/home/sharan/Documents/python/incense_stick_quality_control/runs/opencv_detection/segmentation/defective/segmentation_candidates_adaptive_inverted.png")),
                    visualization
                )

        if primary_stick is not None:
            print(
                f"Primary Stick: "
                f"Area: {primary_stick.area:.2f}, "
                f"Bounding Box: (x: {primary_stick.x}, y: {primary_stick.y}, width: {primary_stick.width}, height: {primary_stick.height}), "
                f"Aspect Ratio: {primary_stick.aspect_ratio:.2f}, "
                f"Image Area Ratio: {primary_stick.image_area_ratio:.4f}"
            )

            stick_mask = create_stick_mask(
                results.get("resized").shape,
                primary_stick.contour
            )

            cv2.imwrite(
                str(Path(f"/home/sharan/Documents/python/incense_stick_quality_control/runs/opencv_detection/segmentation/good/stick_mask.png")),
                stick_mask
            )
        else:
            print("No primary stick found.")



if __name__ == "__main__":
    main()