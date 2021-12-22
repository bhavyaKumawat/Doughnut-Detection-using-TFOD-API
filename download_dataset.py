# Download images from URL files
# How to consume?
# python download_dataset.py --urls <URLs_file> --output <output_directory>

import requests
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True, help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True, help="path to output directory of images")
args = vars(ap.parse_args())

urls = args["urls"]
output = args["output"]
total = 0

rows = open(urls).read().strip().split("\n")

for url in rows:
    try:
        # try to download the image
        r = requests.get(url, timeout=10)
        # save the image to disk
        p = os.path.join(output, "{}.jpg".format(
            str(total).zfill(5)))
        f = open(p, "wb")
        f.write(r.content)
        f.close()
        # update the counter
        print("Downloaded: {}".format(p))
        total += 1
    # handle if any exceptions are thrown during the download process
    except:
        print("Error downloading {}...skipping".format(p))
