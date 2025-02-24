# Automated SEM labelling_ReadMe

## What will this code do

After image acquisition one will save SEM images as shown in the image below. While the banner is great for some Meta data, put not ideal for presentations and publicatoins.

<img src="3033_IBE600W_25min_pat_15.png" width = "500">

For those applications this code ist here. The Scale bar and it's caption on the left side of the banner will be recognized automatically and then transfered on a banner free image. The size of the label background is proportoinal to the image size.

<img src="3033_IBE600W_25min_pat_15_processed.png" width = "500">


The images are saved in .png, .jpg and .svg in a specified folder.



## What do you need

It start s with the image acquisiton, when the images are saved a subset of image has to be saved. This subset does not have a banner and is called ***PlainImages*** by default of the Zeiss SmartSEM software. These images should be kept as a subfolder in the image folder.

The following packages needs to be installed:
opencv-python: 
```
pip install opencv-python
```
easyocr: 
```
pip install easyocr
```

numpy: 
```
pip install numpy
```

pillow: 
```
pip install pillow
```

matplotlib:
```
pip install matplotlib
```

tkinter:
Is included with python's standard library, for Linux use:
```
sudo apt install python3-tk  # For Ubuntu/Debian
sudo dnf install python3-tkinter  # For Fedora
```


## What will come up next

- Improved text recognition
- Drag&Drop of images instead of numbered lst to select from
