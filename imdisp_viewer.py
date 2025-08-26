import os
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Union
import ast


class ImDisp:
    def __init__(self,
                 images: Union[str, np.ndarray, List[Union[np.ndarray, str]]] = None,
                 grid_size: Tuple[int, int] = None,
                 display_range: Tuple[float, float] = None,
                 border: float = 0.01,
                 cmap: str = 'gray',
                 result_table: pd.DataFrame = None,
                 image_ids: List[int] = None):

        self.border = border
        self.cmap = cmap
        self.grid_size = grid_size
        self.display_range = display_range
        self.current_page = 0

        self.images = self._load_images(images)
        self.n_images = len(self.images)
        self.rows, self.cols = self._compute_grid()
        self.images_per_page = self.rows * self.cols
        self.total_pages = math.ceil(self.n_images / self.images_per_page)

        self.result_table = result_table
        self.image_ids = image_ids

        self._init_plot()

    def _load_images(self, images):
        if images is None:
            supported = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            images = [f for f in os.listdir('.') if f.lower().endswith(supported)]

        if isinstance(images, (str, np.ndarray)):
            images = [images]

        loaded = []
        for img in images:
            if isinstance(img, str):
                img_data = np.array(Image.open(img).convert('RGB'))
            elif isinstance(img, np.ndarray):
                img_data = img
            else:
                raise ValueError("Images must be file paths or numpy arrays.")
            loaded.append(img_data)
        return loaded

    def _compute_grid(self):
        if self.grid_size:
            return self.grid_size
        cols = math.ceil(math.sqrt(self.n_images))
        rows = math.ceil(self.n_images / cols)
        return rows, cols

    def _init_plot(self):
        self.fig, self.axs = plt.subplots(self.rows, self.cols, squeeze=False)
        self.fig.subplots_adjust(wspace=self.border, hspace=self.border)
        self._connect_keys()
        self._draw_page()

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        plt.show()

    def _draw_page(self):
        start = self.current_page * self.images_per_page
        end = min(start + self.images_per_page, self.n_images)
        images_this_page = self.images[start:end]

        for i in range(self.rows * self.cols):
            r, c = divmod(i, self.cols)
            ax = self.axs[r][c]
            ax.clear()
            if i < len(images_this_page):
                img = images_this_page[i]
                if self.display_range:
                    ax.imshow(img, vmin=self.display_range[0], vmax=self.display_range[1], cmap=self.cmap)
                else:
                    ax.imshow(img, cmap=self.cmap if img.ndim == 2 else None)
            else:
                ax.imshow(np.zeros((10, 10)), cmap=self.cmap)
            ax.axis('off')
        self.fig.suptitle(f'Page {self.current_page + 1} / {self.total_pages}', fontsize=14)
        self.fig.canvas.draw_idle()

    def _connect_keys(self):
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key in ['right', 'down']:
            self.current_page = (self.current_page + 1) % self.total_pages
            self._draw_page()
        elif event.key in ['left', 'up']:
            self.current_page = (self.current_page - 1) % self.total_pages
            self._draw_page()

    def _on_click(self, event):
        if event.dblclick and event.inaxes:
            ax = event.inaxes
            ax_list = self.axs.flatten().tolist()
            if ax in ax_list:
                index_in_grid = ax_list.index(ax)
                index = self.current_page * self.images_per_page + index_in_grid

                if index < len(self.images):
                    image_id = self.image_ids[index]
                    row = self.result_table[self.result_table['ResultID'] == image_id]
                    if not row.empty:
                        lumen_coords_raw = row['Adjusted_LumenPixels'].values[0]
                        border_coords_raw = row['Adjusted_BorderPixels'].values[0]

                        lumen_fixed = lumen_coords_raw.replace("\n", ",").replace(' ', ', ').replace(',,', ',').replace(', ,', ',').replace('[ ,', '[').replace('[, ', '[').replace(', ,', ',')
                        border_fixed = border_coords_raw.replace("\n", ",").replace(' ', ', ').replace(',,', ',').replace(', ,', ',').replace('[ ,', '[').replace('[, ', '[').replace(', ,', ',')

                        try:
                            lumen_array = np.array(ast.literal_eval(lumen_fixed))
                            border_array = np.array(ast.literal_eval(border_fixed))
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing coordinates: {e}")
                            return

                        img = self.images[index]
                        h, w = img.shape[:2]
                        cx, cy = w // 2, h // 2

                        lumen_coords = [(int(y + cy), int(x + cx)) for x, y in lumen_array]
                        border_coords = [(int(y + cy), int(x + cx)) for x, y in border_array]

                        fig_overlay, ax_overlay = plt.subplots()
                        ax_overlay.imshow(img, cmap=self.cmap if img.ndim == 2 else None)

                        overlay = np.zeros((*img.shape[:2], 4), dtype=np.float32)

                        for y, x in lumen_coords:
                            if 0 <= y < h and 0 <= x < w:
                                overlay[y, x] = [0, 1, 0, 0.4]

                        for y, x in border_coords:
                            if 0 <= y < h and 0 <= x < w:
                                overlay[y, x] = [1, 0, 0, 0.4]

                        ax_overlay.imshow(overlay)
                        ax_overlay.set_title(f'Lumen & Border overlay for obj {image_id}')
                        ax_overlay.axis('off')
                        fig_overlay.tight_layout()

                        # Enable zoom and pan
                        fig_overlay.canvas.toolbar_visible = True
                        fig_overlay.canvas.header_visible = False
                        fig_overlay.canvas.footer_visible = False
                        fig_overlay.canvas.resizable = True

                        plt.show()


def imdisp(images=None, grid_size=None, display_range=None, border=0.01, cmap='gray', result_table=None, image_ids=None):
    ImDisp(
        images=images,
        grid_size=grid_size,
        display_range=display_range,
        border=border,
        cmap=cmap,
        result_table=result_table,
        image_ids=image_ids
    )
