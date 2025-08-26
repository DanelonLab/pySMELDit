# SMELDit 🧪 — Smart Liposome Image Detection & Analysis Tool

**SMELDit** is an interactive Python pipeline for analyzing fluorescence microscopy images of liposomes. It provides tools to select ROIs, detect liposomes, extract spatial and pixel data, and save cropped liposome images — all organized per biological sample.

---

## 🚀 Quick Start

### 1. Install Requirements

Make sure you have Python 3.8+ and the required packages installed:

`pip install numpy pandas imageio matplotlib`

### 2. Run the Data Processor

To run the main analysis pipeline:

`python process_data.py`

You'll be prompted to:
- Enter a sample name e.g. `TestSample`
- Select one or more *TIFF* image files

### 3. Run SMELDit and visualize your liposomes

Run `python SMELDit.py`

You'll then be able to:
- Load your sample (e.g. `Recognized_Liposomes/TestSample/data_file.csv`)
- Select what quantities to plot against each other on the x- and y-axis
- `Draw ROI` to draw a polygon representing your region of interest
- `Import ROI`/`Export ROI` to save and retrieve your ROIs as `.npz` files
- `Save Selection`/`Load Selection` to save all data points contained within your ROI to a new `.csv` data file and load such data files

## 📁 Output Structure
Output is organized by sample name inside the `Recognized_Liposomes/` folder:

    Recognized_Liposomes/
    └── sample_name/
        ├── data_file.csv
        ├── crop_ID_00001.tif
        ├── crop_ID_00002.tif
        └── ...

Each cropped image is a single liposome region, and data_file.csv contains all metadata.

## 📊 What’s Inside data_file.csv?

Each row represents one detected liposome, with columns like:

- `ResultID` — Unique ID across all images in this sample

- `Centroids` — X, Y coordinates of the liposome center

- `Adjusted_LumenPixels, Adjusted_BorderPixels` — Cropped-relative coordinates of pixels belonging to the lumen and the liposome membrane

- `SourceImage` — The original TIFF file it came from

## 🖼 Image Format
Input TIFFs must be 3-channel grayscale images, where:

- Channel 1: `C1` (e.g., dsGreen)

- Channel 2: `C2` (e.g., mCherry)

- Channel 3: `C3` (e.g., Cy5)

Both (3, H, W) and (H, W, 3) formats are supported.

## 📦 Files Overview
| File | Purpose |
| --- | ----------- |
| `process_data.py` | Main Image Analysis tool: prompts for sample name & images |
| `SMELDit.py` | GUI-based ROI selection & liposome collage visualization |
| `image_analysis_module.py` | Core logic for detection and cropping |
| `imdisp_viewer.py` | Image display tools

## 🧠 Customization
- ROI drawing and export lives in `SMELDit` (GUI).

- Modify `image_analysis()` in `image_analysis_module.py` to tweak how liposomes are detected.

- The script automatically avoids ID collisions when appending across multiple TIFFs.

## 📬 Contact
For questions, issues, or collaborations, feel free to contact:

Corresponding Author

danelon@insa-toulouse.fr