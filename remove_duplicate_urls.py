# Delete duplicate URLs
# How to consume?
# python remove_duplicate_urls.py --dir <dataset>

import os
import glob
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Directory Path")
args = vars(ap.parse_args())

dir_ = args["dir"]
new_dir = "unique_{}".format(dir_)
os.mkdir(new_dir)

final_urls = {}
duplicates = 0
files = glob.glob(os.path.join(".", dir_,"*" ))
for file in files:
    text = open(file)
    urls = text.read().strip().split()
    unique_urls = []
    for url in urls:
        if url not in final_urls:
             final_urls[url] = file.split(".")[:-1]
             unique_urls.append(url)
        else:
            duplicates+=1 
    with open( file.replace(dir_, new_dir), 'w') as f:
        for url in unique_urls:
            f.write("%s\n" % url)
            
print("Found {} duplicates!!! \nDeleting...".format(duplicates))
