# environment:
python         3.9 <br>
opencv-python  4.5.5.64 <br>
torch          1.9.0+cu111 <br>
scikit-image   0.18.2 <br>
Pillow         9.1.0 <br>
apex **（ warning: If you want to install "apex", please find it in https://github.com/NVIDIA/apex to download and install.）**

# run:
run main.py to start the progrem <br>


## Train:
change 'mood' to "train" in config.py


## Test:
change 'mood' to "test" in config.py

# tips
This program contains the perspective transformation model the corners are in the file "cut_corners",if you don't want the perspective transformation model works please change the corners in "cut_corners" all to "(0,0)
