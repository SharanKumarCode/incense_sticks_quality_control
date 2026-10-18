from pathlib import Path
import cv2

from services.opencv_detection.preprocessing import ImagePreprocessor
from services.opencv_detection.image_transform_helper import rotate_mask_horizontal
from services.opencv_detection.segmentation import find_stick_candidates, select_primary_stick, create_stick_mask
from services.opencv_detection.geometry import extract_geometry
from services.opencv_detection.visualization import draw_geometry, draw_straightness, draw_breakage, draw_width_profile, draw_width_anomalies, draw_shape_irregularity
from services.opencv_detection.straightness import analyse_straightness
from services.opencv_detection.structural import analyse_breakage, calculate_width_profile, analyse_width_anomalies, analyse_shape_irregularity

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

    stick_mask = create_stick_mask(
        results.get("resized").shape,
        primary_stick.contour
    )

    rotated_stick_mask, _ = rotate_mask_horizontal(
        stick_mask,
        geometry.orientation_deg
    )

    straightness = analyse_straightness(
        contour=primary_stick.contour,
        image_shape=results["resized"].shape,
        orientation_deg=geometry.orientation_deg,
        length_px=geometry.length_px
    )

    breakage = analyse_breakage(
        binary_mask=results.get("adaptive_inverted"),
        geometry=geometry
    )

    width_profile = calculate_width_profile(rotated_stick_mask=rotated_stick_mask)

    width_anomalies = analyse_width_anomalies(width_profile)

    shape_irregularities = analyse_shape_irregularity(primary_stick.contour)

    # Step 3B
    output_path = OUTPUT_DIR / f"{IMAGE_PATH.stem}_geometry.jpg"

    straightness_output = OUTPUT_DIR / f"{IMAGE_PATH.stem}_straightness.jpg"

    breakage_output = OUTPUT_DIR / f"{IMAGE_PATH.stem}_breakage.jpg"

    width_profile_output = OUTPUT_DIR / f"{IMAGE_PATH.stem}_width_profile.jpg"

    width_anomalies_output = OUTPUT_DIR / f"{IMAGE_PATH.stem}_width_anomalies.jpg"

    shape_irregularity_output = OUTPUT_DIR / f"{IMAGE_PATH.stem}_shape_irregularity.jpg"

    visualization = draw_geometry(results["resized"], primary_stick.contour, geometry)

    cv2.imwrite(
        str(output_path),
        visualization
        )

    straighness_visualisation = draw_straightness(straightness)

    cv2.imwrite(
            str(straightness_output),
            straighness_visualisation
            )

    breakage_visualisation = draw_breakage(breakage)

    cv2.imwrite(
                str(breakage_output),
                breakage_visualisation
                )

    width_profile_visualisation = draw_width_profile(rotated_stick_mask, width_profile)
    
    cv2.imwrite(
                str(width_profile_output),
                width_profile_visualisation
                )

    width_anomalies_visualisation = draw_width_anomalies(rotated_stick_mask, width_profile, width_anomalies)
        
    cv2.imwrite(
                str(width_anomalies_output),
                width_anomalies_visualisation
                )

    shape_irregularity_visualisation = draw_shape_irregularity(results["resized"], primary_stick.contour, shape_irregularities)
            
    cv2.imwrite(
                str(shape_irregularity_output),
                shape_irregularity_visualisation
                )
    

    print(f"Saved : {output_path}")

    print(f"Saved Straightness diagnostic: {straightness_output}")

    print(f"Saved Breakage diagnostic: {breakage_output}")

    print(f"Saved Width Profile diagnostic: {width_profile_output}")

    print(f"Saved Width Anomaly diagnostic: {width_anomalies_output}")

    print(f"Saved Shape Irregularity diagnostic: {shape_irregularity_output}")



if __name__ == "__main__":
    main()