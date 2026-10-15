import json

from services.opencv_detection.detector import OpenCVDetector

from pathlib import Path

def main():
    detector = OpenCVDetector()
    input_folder_path = Path("/home/sharan/Documents/python/incense_stick_quality_control/input/")
    output_folder_path = Path("/home/sharan/Documents/python/incense_stick_quality_control/output/")

    results_file_path = output_folder_path / "results.json"

    bad_sticks_folder_path = input_folder_path / "damaged"
    good_sticks_folder_path = input_folder_path / "good"

    if not (output_folder_path.exists() or input_folder_path.exists() or bad_sticks_folder_path.exists() or good_sticks_folder_path.exists()):
        print("One or more required folders do not exist.")
        return

    bad_sticks_files = [f for f in bad_sticks_folder_path.iterdir() if f.is_file()]
    good_sticks_files = [f for f in good_sticks_folder_path.iterdir() if f.is_file()]

    results = {
        "bad_sticks": [],
        "good_sticks": []
    }

    total_bad_sticks = len(bad_sticks_files)
    total_good_sticks = len(good_sticks_files)

    for i, file in enumerate(bad_sticks_files, start=1):
        result = detector.detect(str(file))
        results["bad_sticks"].append({
            "file": str(file),
            "is_defective": result.is_defective,
            "detected_defects": result.detected_defects,
            "measurements": result.measurements
        })
        print(f"Processed {i}/{total_bad_sticks}: {file} - Defective: {result.is_defective}")
        

    for i, file in enumerate(good_sticks_files, start=1):
        result = detector.detect(str(file))
        results["good_sticks"].append({
            "file": str(file),
            "is_defective": result.is_defective,
            "detected_defects": result.detected_defects,
            "measurements": result.measurements
        })
        print(f"Processed {i}/{total_good_sticks}: {file} - Defective: {result.is_defective}")
    
    # Save results to JSON file
    with open(results_file_path, "w") as f:
        json.dump(results, f)
        print(f"Results saved to {results_file_path}")


if  __name__ == "__main__":
    main()