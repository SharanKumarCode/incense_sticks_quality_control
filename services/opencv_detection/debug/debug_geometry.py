from pathlib import Path
import cv2

from services.opencv_detection.preprocessing import ImagePreprocessor
from services.opencv_detection.segmentation import find_stick_candidates, select_primary_stick
from services.opencv_detection.geometry import extract_geometry
from services.opencv_detection.visualization import draw_geometry

IMAGE_PATH = Path("/home/sharan/Documents/python/incense_stick_quality_control/dataset/train/good/20260728_064445.jpg")

# IMAGE_PATH = Path("/home/sharan/Documents/python/incense_stick_quality_control/dataset/train/defective/IMG-20260728-WA0006.jpg")

OUTPUT_DIR = Path("/home/sharan/Documents/python/incense_stick_quality_control/runs/opencv_detection/geometry/good")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1
    preprocessor = ImagePreprocessor()
    results = preprocessor.preprocess(IMAGE_PATH)

    # Step 2
    mask = results.get("adaptive_inverted")

    if mask is None:
        print(f"No masks found")
        return
    
    candidates = find_stick_candidates(mask)
    primary_stick = select_primary_stick(candidates)

    if primary_stick is None:
        print("No stick detected")
        return

    # Step 3A
    geometry = extract_geometry(primary_stick.contour)

    # Step 3B
    visualization = draw_geometry(results["resized"], primary_stick.contour, geometry)

    output_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_geometry.jpg"

    cv2.imwrite(
        str(output_path),
        visualization
        )

    print(f"Saved : {output_path}")

    print(f"Geometry : {geometry}")



if __name__ == "__main__":
    main()