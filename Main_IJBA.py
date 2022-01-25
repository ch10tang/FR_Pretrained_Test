from Model.model_irse import IR_50
from Model.model_lightcnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from Model.resnet50_ft_dims_2048 import resnet50_ft
import torch
import argparse
import pandas as pd
import os
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from numpy.linalg import norm
from util.PoseSelect import PoseSelect
from util.LoadPretrained import LoadPretrained
from util.DataLoader import FaceIdPoseDataset
from util.ConcatPath import ConcatPath
from util.InputSize_Select import Transform_Select
# from util.Validate_MPIE import Validate_MPIE
from util.Validation_IJBA import Validation_IJBA




if __name__=="__main__":
    # ImageName = 'SA_FNM_MB16_CAISAwoAlign_Wild_LP_Manually_75_82_Fea3500_FaceOnly_29_0'
    parser = argparse.ArgumentParser(description='Eval_SOTA_Model')
    # learning & saving parameters
    parser.add_argument('-data-place', type=str, default='F:/Database', help='prepared data path to run program')
    parser.add_argument('-csv-file', type=str, default='../DataList/IJB-A_FOCropped_250_250_84.csv', help='csv file to load image for training')
    parser.add_argument('-model-select', type=str, default='VGGFace2', help='Model Select')
    parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 8]')
    # Evaluation options
    parser.add_argument('-Save-Features', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-Eval-CFP', action='store_true', default=False, help='enable the gpu')

    args = parser.parse_args()


    BACKBONE_DICT = {'IR-50': IR_50(112),
                 'Light_CNN_9': LightCNN_9Layers(),
                 'Light_CNN_29': LightCNN_29Layers(),
                 'Light_CNN_29_v2': LightCNN_29Layers_v2(),
                 'VGGFace2': resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth')
    }
    BACKBONE = BACKBONE_DICT[args.model_select]
    Model = LoadPretrained(BACKBONE, args)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Model = Model.to(device)

    _ = Validation_IJBA(Model, 'Temp_IJBA_Feature/IR50', 2, 100, device, args)














