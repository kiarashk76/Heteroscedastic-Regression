import glob
from PIL import Image

# file_path = input("Please enter the file path to animate: ")
# filepaths
def animate(fp_in, fp_out):
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
            save_all=True, duration=200, loop=0)