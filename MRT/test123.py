import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2

bild = nib.load('sub10002_s0.nii.gz').get_fdata()
bildgroesse = bild.shape
fig, ax = plt.subplots()
ims = []
image_size = bildgroesse[0] * bildgroesse [1]

resize = 200

for z in range(150,bildgroesse[2]):
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    for y in range(bildgroesse[0]):
        if np.sum(bild[y,:,z]) > 0 and y1 == 0:
            y1 = y
        if np.sum(bild[bildgroesse[0]-y-1,:,z]) == 0 and np.sum(bild[bildgroesse[0]-y-2,:,z]) > 0:
            y2 = bildgroesse[0]-y-2
    for x in range(bildgroesse[1]):
        if np.sum(bild[:,x,z]) > 0 and x1 == 0:
            x1 = x
        if np.sum(bild[:,bildgroesse[1]-x-1,z]) == 0 and np.sum(bild[:,bildgroesse[1]-x-2,z]) > 0:
            x2 = bildgroesse[1]-x-2

    rimage = cv2.resize(bild[y1:y2,x1:x2,z],(resize,resize),interpolation=cv2.INTER_LINEAR)
    plt.imshow(rimage)
    plt.title(f'Schicht {z+1}')
    plt.show(block=False)
    plt.pause(2)
    plt.close()


