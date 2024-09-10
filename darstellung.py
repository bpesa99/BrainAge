import os
import gzip
import shutil
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


pinfo = pd.read_csv('info.csv')
path = 'MRT'
personen = os.listdir((path))

i = 0
while i < len(personen):
    bildpath = path+'/{0}'.format(personen[i])
    bilder = os.listdir(bildpath) #Anzahl aufgenommener Bilder bzw. teilgenommener Surveys
    j = 0
    while j < len(bilder):
        #entpacken des Bildes
        with gzip.open(bildpath+'/{Bildnummer}/anat/{ID}_{Bildnummer}_T1w.nii.gz'.format(Bildnummer=bilder[j], ID=personen[i]), 'rb') as f_in:
            with open(bildpath+'/{Bildnummer}/anat/{ID}_{Bildnummer}_T1w.nii'.format(Bildnummer=bilder[j], ID=personen[i]), 'wb') as f_out: shutil.copyfileobj(f_in,f_out)
                
        bild = nib.load(bildpath+'/{Bildnummer}/anat/{ID}_{Bildnummer}_T1w.nii'.format(Bildnummer=bilder[j], ID=personen[i])).get_fdata()
        bildgroesse = bild.shape

        survey = int(bilder[j][-1:])
        match survey: #richtiges Alter zum passenden Survey
            case 0:
                age = pinfo.iloc[i,4]
            case 2:
                age = pinfo.iloc[i, 3]
            case 4:
                age = pinfo.iloc[i, 7]
            case 6:
                age = pinfo.iloc[i, 8]
        print('Age = {0}'.format(age))
        
        fig, ax = plt.subplots()
        ims = []
        for x in range(bildgroesse[2]): #Bildausgabe als Animation
            im = ax.imshow(bild[:,:,x], animated=True)
            if x == 0:
                ax.imshow(bild[:,:,0])
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)
        plt.show(block=False)
        plt.pause(16)
        plt.close()

        j = j + 1

    i = i + 1

