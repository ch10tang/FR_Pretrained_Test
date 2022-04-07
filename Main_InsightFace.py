from Model.model_irse import IR_50
# from Model.model_irse_eval import IR_50 as ArcFace_AVL_LargePose
from Model.model_irse_eval import IR_50 as ArcFace_AVL_LargePose
from Model.model_resnet import ResNet_50 as ArcFace_LargePose
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
import winsound






if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Eval_SOTA_Model')
    # learning & saving parameters
    parser.add_argument('-data-place', type=str, default='F:/Database', help='prepared data path to run program')
    parser.add_argument('-csv-file', type=str, default='../DataList/IJBC_InsightFace.csv', help='csv file to load image for training')
    #parser.add_argument('-csv-file', type=str, default='../DataList/IJB-A_FOCropped_250_250_84.csv', help='csv file to load image for training')
    parser.add_argument('-model-select', type=str, default='VGGFace2', help='Model Select')
    parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 8]')
    # Evaluation options
    # parser.add_argument('-generate-place', type=str, default='D:/04_FaceEvaluation/Experiment_IJBA_Mean_v02/Feature/SA_L1_FNM_MB4_Fea3500_190_Illum_Sym001_All_w_AllGP_l1_3_0', help='prepared data path to run program')
    parser.add_argument('-generate-place', type=str, default='../../04_FaceEvaluation/Experiment_IJBC_Turtorial/_Features/VGGFace2_LooseCrop', help='prepared data path to run program')
    parser.add_argument('-Save-Features', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-Eval-CFP', action='store_true', default=False, help='enable the gpu')

    args = parser.parse_args()


    BACKBONE_DICT = {'IR-50': IR_50(112),
                 'Light_CNN_9': LightCNN_9Layers(),
                 'Light_CNN_29': LightCNN_29Layers(),
                 'Light_CNN_29_v2': LightCNN_29Layers_v2(),
                 'VGGFace2': resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth'),
    }
    BACKBONE = BACKBONE_DICT[args.model_select]
    Model = LoadPretrained(BACKBONE, args)

    save_dir = '{}'.format(args.generate_place)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    if not args.Save_Features and not args.Eval_CFP:
        print('Please select valid option for saving features (args.Save_Features) or evalating on CFP (args.Eval_CFP)')
        print('Loading the default setting (Save_Features)')
        args.Eval_CFP = True


    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Model = Model.to(device)
    Model.eval()

    # Load augmented data
    transforms = Transform_Select(args)
    transformed_dataset = FaceIdPoseDataset(args.csv_file, args.data_place, transform=transforms)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False)  # , num_workers=6)

    Features_List = {'Subject':[], 'Pose':[], 'ImgNum':[], 'Features':[]}

    count = 0
    minibatch_size = args.batch_size
    Frame = {}
    if args.model_select.startswith('IR-50') or args.model_select.startswith('ArcFace'): Frame = np.zeros((469375, 512))
    elif args.model_select.startswith('Light_CNN'): Frame = np.zeros((469375, 256))
    elif args.model_select.startswith('VGGFace2'): Frame = np.zeros((469375, 2048))
    
    for i, batch_data in enumerate(dataloader):
   
        if args.model_select == 'VGGFace2': batch_image = (batch_data[0]*255).to(device)
        else: batch_image = batch_data[0].to(device)

        _ = Model(batch_image)
        try: Model_Feature = Model.feature
        except: Model_Feature = Model.module.feature

        if len(Model_Feature.shape) > 2:
            Model_Feature = Model_Feature.view(Model_Feature.size(0), -1)
        features = (Model_Feature.data).cpu().numpy()
        batchImageName = batch_data[1]

        if args.Save_Features:
            for feature, ImgName in zip(features, batchImageName):
                Frame[int(ImgName.split('.')[0]) - 1, :] = feature
            count += minibatch_size
            print("Finish Processing {} images...".format(count))

    np.save('{}/Features.npy'.format(save_dir), Frame)
    