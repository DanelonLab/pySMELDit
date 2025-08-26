import os
import tkinter as tk
from tkinter import filedialog, StringVar, Label, Button
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon
from imdisp_viewer import imdisp

class DataPlotter:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Histogram with ROI")
        self.df = None
        self.canvas = None
        self.selector = None
        self.roi_patch = None
        self.mask = None
        self.x_data = None
        self.y_data = None

        self.metric_names = {
            "Area": "Area",
            "dsGreen": "dsGreen intensity",
            "dsGreen_var": "dsGreen var",
            "mCherry": "mCherry intensity",
            "mCherry_var": "mCherry var",
            "Cy5": "Cy5 intensity",
        }

        self.x_var = StringVar()
        self.y_var = StringVar()

        self.setup_widgets()

    def setup_widgets(self):
        Button(self.root, text="Load Data", command=self.load_data).pack()

        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack()

        self.dropdown_frame = tk.Frame(self.root)
        self.dropdown_frame.pack()

        Label(self.dropdown_frame, text="X-axis:").grid(row=0, column=0)
        self.x_menu = None

        Label(self.dropdown_frame, text="Y-axis:").grid(row=1, column=0)
        self.y_menu = None

        Button(self.root, text="Draw ROI", command=self.activate_roi).pack()
        Button(self.root, text="Save Selection", command=self.save_roi).pack(side='left')
        Button(self.root, text="Load Selection", command=self.load_roi_file).pack(side='left')
        Button(self.root, text="Export ROI", command=self.export_roi).pack(side='right')
        Button(self.root, text="Import ROI", command=self.import_roi).pack(side='right')
        Button(self.root, text="Show Images in ROI", command=self.display_roi_images).pack(side='right')

        self.roi_text = Label(self.root, text="Points in ROI: 0")
        self.roi_text.pack()

    def load_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        self.data_file_path = file_path
        self.data_folder = os.path.dirname(file_path)
        print("Loading data from", file_path)
        self.df = pd.read_csv(file_path)

        columns = list(self.metric_names.keys())
        self.x_var.set(columns[0])
        self.y_var.set(columns[1])

        if self.x_menu:
            self.x_menu.destroy()
        if self.y_menu:
            self.y_menu.destroy()

        self.x_menu = tk.OptionMenu(self.dropdown_frame, self.x_var, *columns, command=lambda _: self.update_plot())
        self.x_menu.grid(row=0, column=1)

        self.y_menu = tk.OptionMenu(self.dropdown_frame, self.y_var, *columns, command=lambda _: self.update_plot())
        self.y_menu.grid(row=1, column=1)

        self.update_plot()

    def display_roi_images(self):
        if self.mask is None or self.df is None:
            print("No ROI selected.")
            return

        folder = getattr(self, 'data_folder', None)
        if not folder or not os.path.isdir(folder):
            folder = filedialog.askdirectory(title="Select Folder with Cropped Images")
            if not folder:
                return

        # Collect ResultIDs for data points in ROI
        roi_ids = self.df.loc[self.mask, "ResultID"].astype(int)

        # Reconstruct filenames and collect matched ResultIDs
        filenames = []
        image_ids = []  # This will store matched ResultIDs in the same order as filenames

        for obj_id in roi_ids:
            pattern = f"_obj{obj_id:05d}.tif"
            for fname in os.listdir(folder):
                if fname.endswith(pattern):
                    filenames.append(os.path.join(folder, fname))
                    image_ids.append(obj_id)
                    break  # Only the first match is needed

        if filenames:
            #print("Calling imdisp with:", filenames)
            imdisp(
                images=filenames,
                grid_size=(8, 8),
                result_table=self.df,
                image_ids=image_ids
            )
        else:
            print("No matching images found for ROI.")


    def update_plot(self, from_roi=False):
        x = self.x_var.get()
        y = self.y_var.get()

        if self.df is None:
            return

        data = self.df.dropna(subset=[x, y])

        x_data = np.log10(data[x] + 0.1)
        y_data = np.log10(data[y] + 0.1)

        self.x_data = x_data.values
        self.y_data = y_data.values

        fig = plt.Figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5), wspace=0.05, hspace=0.05)
        ax_histx = fig.add_subplot(gs[0, 0])
        ax_main = fig.add_subplot(gs[1, 0])
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])

        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)

        counts, xedges, yedges, im = ax_main.hist2d(
            self.x_data, self.y_data,
            bins=100,
            norm=plt.cm.colors.LogNorm(),
            cmap="viridis"
        )

        ax_histx.hist(self.x_data, bins=100, color='gray')
        ax_histx.set_yscale('log')
        ax_histx.set_ylabel("Count")

        ax_histy.hist(self.y_data, bins=100, orientation='horizontal', color='gray')
        ax_histy.set_xscale('log')
        ax_histy.set_xlabel("Count")

        ax_main.set_xlabel(self.metric_names[x])
        ax_main.set_ylabel(self.metric_names[y], labelpad=10)
        ax_main.set_xlim((np.amin(x_data)/1.2,np.amax(x_data)*1.2))
        ax_main.set_ylim((np.amin(y_data)/1.2,np.amax(y_data)*1.2))
        ax_main.set_aspect(1.0, adjustable='datalim')

        fig.colorbar(im, cax=cbar_ax)
        fig.tight_layout()
        self.ax_main = ax_main

        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def activate_roi(self):
        if self.selector:
            self.selector.disconnect_events()
            self.selector = None

        if self.roi_patch:
            self.roi_patch.remove()
            self.roi_patch = None
            self.canvas.draw()

        self.selector = PolygonSelector(
            self.ax_main,
            self.on_select,
            useblit=True,
            props=dict(color='red', linewidth=2, alpha=0.5)
        )

    def on_select(self, verts):
        self.selector._xs, self.selector._ys = [0], [0]
        self.selector.set_visible(False)
        path = Path(verts)
        points = np.vstack((self.x_data, self.y_data)).T
        self.mask = path.contains_points(points)
        count = np.sum(self.mask)
        self.roi_text.config(text=f"Points in ROI: {count}")

        if self.roi_patch:
            #print('deleting ROI')
            try:
                self.roi_patch.remove()

            except ValueError:
                pass
            self.roi_patch = None

        self.roi_patch = Polygon(verts, closed=True, edgecolor='red', facecolor='none', linewidth=2, alpha=0.5)
        self.ax_main.add_patch(self.roi_patch)
        self.canvas.draw()

    def save_roi(self):
        if self.mask is None or self.df is None:
            return
        roi_df = self.df[self.mask]
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if save_path:
            roi_df.to_csv(save_path, index=False)
            print(f"Saved ROI data to {save_path}")

    def load_roi_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.df = pd.read_csv(file_path)
            self.update_plot(from_roi=True)
    
    def export_roi(self):
        if self.roi_patch is None or self.x_var.get() == "" or self.y_var.get() == "":
            print("No ROI to export.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy Archive", "*.npz")])
        if not save_path:
            return

        verts = self.roi_patch.get_xy()
        np.savez(
            save_path,
            verts=verts,
            x_axis=self.x_var.get(),
            y_axis=self.y_var.get()
        )
        print(f"ROI exported to {save_path}")

    def import_roi(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy Archive", "*.npz")])
        if not file_path:
            return

        try:
            data = np.load(file_path, allow_pickle=True)
            verts = data["verts"]
            x_axis = str(data["x_axis"])
            y_axis = str(data["y_axis"])
        except Exception as e:
            print(f"Failed to load ROI file: {e}")
            return

        # Set axis selections
        self.x_var.set(x_axis)
        self.y_var.set(y_axis)
        self.update_plot()

        # Remove existing ROI if any
        if self.roi_patch:
            self.roi_patch.remove()
            self.roi_patch = None

        # Draw new ROI patch
        self.roi_patch = Polygon(verts, closed=True, edgecolor='red', facecolor='none', linewidth=2, alpha=0.5)
        self.ax_main.add_patch(self.roi_patch)

        # Recalculate mask
        path = Path(verts)
        points = np.vstack((self.x_data, self.y_data)).T
        self.mask = path.contains_points(points)
        count = np.sum(self.mask)
        self.roi_text.config(text=f"Points in ROI: {count}")

        self.canvas.draw()
        print("ROI imported successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataPlotter(root)
    root.mainloop()

