# resize an image while maintaining the aspect ratio and add padding
# How to consume?
# python resize_dataset.py --dir <dataset>


import os
import argparse
from PIL import Image, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Directory Path")
args = vars(ap.parse_args())

dir_ = args["dir"]
new_dir = os.path.join(os.getcwd(), 'Resized_'+dir_)
os.makedirs(new_dir, exist_ok=True)

    
images = os.listdir(dir_)

for img in images:
	image = Image.open(os.path.join(dir_, img))
	newsize = (1024, 1024)
	image.thumbnail(newsize, Image.ANTIALIAS)
	img_path = os.path.join(new_dir, img)

	delta_width = newsize[0] - image.size[0]
	delta_height = newsize[1] - image.size[1]
	pad_width = delta_width // 2
	pad_height = delta_height // 2
	padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
	pad_img = ImageOps.expand(image, padding)   

	pad_img.save(img_path)
        
print("Created resized dataset!")







