import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from image_analysis_module import image_analysis, PolygonSelector, crop_and_save_liposomes
import pickle
import os
from tkinter import filedialog, Tk

def load_multichannel_tiff(filepath):
    """
    Load a 3-channel TIFF image and return three grayscale numpy arrays: C1, C2, and C3.
    """
    img = imageio.imread(filepath)
    
    if img.ndim == 3:
        if img.shape[0] == 3:
            C1, C2, C3 = img[0], img[1], img[2]
        elif img.shape[2] == 3:
            C1, C2, C3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        else:
            raise ValueError(f"Expected 3-channel image. Got shape: {img.shape}")
    else:
        raise ValueError(f"Expected 3D image. Got shape: {img.shape}")
    
    return C1, C2, C3

def main():
    # Prompt for sample name
    sample_name = input("Enter sample name: ").strip()
    if not sample_name:
        print("⚠️ No sample name provided. Exiting.")
        return

    # Setup output folder
    base_folder = "Recognized_Liposomes"
    output_folder = os.path.join(base_folder, sample_name)
    os.makedirs(output_folder, exist_ok=True)

    # File selection
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select TIFF Image Files",
        filetypes=[("TIFF files", "*.tif *.tiff")]
    )

    if not file_paths:
        print("No files selected.")
        return

    CropSize = 30
    all_results = []

    # Check for existing data to continue ID counting
    data_file_path = os.path.join(output_folder, "data_file.csv")
    if os.path.exists(data_file_path):
        existing_df = pd.read_csv(data_file_path)
        global_id_offset = existing_df["ResultID"].max() + 1
    else:
        global_id_offset = 0

    for i, filepath in enumerate(file_paths, 1):
        print(f"\n🔄 Processing file {i}/{len(file_paths)}: {os.path.basename(filepath)}")

        I_dsGreen, I_mCherry, I_Cy5 = load_multichannel_tiff(filepath)
        results = image_analysis(I_Cy5, I_mCherry, I_dsGreen)

        if len(results["ResultID"])==0:  # Skip files with no detected liposomes
            print(f"⚠️ No liposomes found in {filepath}, skipping.")
            continue

        centroids = results["Centroids"]
        original_ids = results["ResultID"]

        # Offset ResultIDs
        new_ids = [oid + global_id_offset for oid in original_ids]
        results["ResultID"] = new_ids
        global_id_offset = max(new_ids) + 1

        LumenPixels = results["LumenPixels"]
        BorderPixels = results["BorderPixels"]

        adjusted_LumenPixels, adjusted_BorderPixels = crop_and_save_liposomes(
            I_Cy5, I_mCherry, I_dsGreen,
            centroids, CropSize, output_folder, sample_name,
            new_ids, LumenPixels, BorderPixels
        )

        centroids_list = [(centroids[0, j], centroids[1, j]) for j in range(centroids.shape[1])]
        results["Centroids"] = centroids_list
        results["Adjusted_LumenPixels"] = adjusted_LumenPixels
        results["Adjusted_BorderPixels"] = adjusted_BorderPixels
        results["SourceImage"] = os.path.basename(filepath)

        df = pd.DataFrame(results)
        all_results.append(df)

    if not all_results:
        print("❌ No data collected. Exiting.")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    # Append to existing or save new data file
    if os.path.exists(data_file_path):
        existing_df = pd.read_csv(data_file_path)
        final_df = pd.concat([existing_df, final_df], ignore_index=True)

    final_df.to_csv(data_file_path, index=False)
    print(f"\n✅ All done. Results saved to: {data_file_path}")


if __name__ == "__main__":
    main()
