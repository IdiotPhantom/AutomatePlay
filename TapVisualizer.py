import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from setting import SCREEN_HEIGHT, SCREEN_WIDTH


class TapVisualizer:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.all_taps = []
        self.all_patches = []

        # Create figure and axes once here
        self.fig, self.ax = plt.subplots()

        if image_path is not None:
            self.open_image(image_path)
        else:
            # No image yet, just show empty axes
            plt.axis('off')
            plt.ion()
            plt.show()

    def open_image(self, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)

        # Clear axes before showing image (in case called again)
        self.ax.clear()
        self.ax.imshow(self.image)
        plt.axis('off')

        # Redraw and interactive mode on
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.ion()
        plt.show()

    def change_image(self, new_image_path):
        self.image_path = new_image_path
        self.image = Image.open(new_image_path)

        # Clear previous figure content
        self.ax.clear()
        self.ax.imshow(self.image)
        plt.axis('off')

        # Clear old taps and patches
        self.all_taps.clear()
        self.all_patches.clear()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_tap(self, x, y, radius=5, color='red'):
        self.all_taps.append((x, y))

        # Create the new tap circle
        circ = patches.Circle(
            (x, y), radius=radius, color=color, fill=True, alpha=0.6)
        patch = self.ax.add_patch(circ)

        # Add to patch list
        self.all_patches.append(patch)

        # Remove old patches if exceeding 50
        if len(self.all_patches) > 50:
            old_patch = self.all_patches.pop(0)
            old_patch.remove()

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def draw_text(self, text, color='yellow', fontsize=30):
        x = 0
        y = 0
        self.ax.text(x, y, text,
                     color=color,
                     fontsize=fontsize,
                     fontweight='bold',
                     bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def keep_open(self):
        plt.ioff()  # Turn OFF interactive mode
        plt.show()  # This will now block
