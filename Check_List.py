import numpy as np
import pandas as pd
import os
import shutil

Img_Path = [
            'F:\Database\IJBA\IJB-A_AVLFOCropped_250_250_84_v02',
            #'GeneratedImages/VGG080_ft_9_0',
            #'GeneratedImages/VGG120_ft_9_0',
            ]

for ImgPath in Img_Path:
    ct = 0
    for idx, (roots, dirs, files) in enumerate(os.walk(ImgPath)):
        for file in files:
            tmp = file.split('.')
            OriginalPath = '{}/{}'.format(roots, file)
            SavePath = '{}/{}.{}'.format(roots, tmp[0], tmp[-1])
            ct += 1
            if OriginalPath==SavePath:
                continue
            else:
                shutil.copy(OriginalPath, SavePath)
        print('{}, {}/25790'.format(ImgPath, ct))


