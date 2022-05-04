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
# from util.Validation_MPIE import Validation_MPIE



if __name__=="__main__":
    Model_Name = 'VGGFace2_TypeP_Sub4979_Img74522_60_92_WarpAffine7refs_Flip'
    Test_Path = 'P4_Style_mdfr_ID_Ep(Warp7refs)_Flip_Map8_ID1_Adv1_IDDis'
    steps = '190000'
    # File_Name = 'Test10'
    parser = argparse.ArgumentParser(description='Eval_SOTA_Model')
    # learning & saving parameters
    parser.add_argument('-data-place', type=str, default='D:/04_FaceEvaluation\Experiment_IJBA_v05\_Generated_Image/{}/{}'.format(Test_Path, steps))
    # parser.add_argument('-data-place', type=str, default='D:/Josh3/1. POFR_Experiments/PON/Test_List/VGGFace2_mdfr/{}'.format(Test_Path))
    parser.add_argument('-csv-file', type=str, default='../DataList/VGGFace2_60_90.csv')
    parser.add_argument('-model-select', type=str, default='IR-50', help='Model Select')
    parser.add_argument('-checkpoints', type=str, default='./Pretrained/{}/Backbone_IR_50_Epoch_90.pth'.format(Model_Name), help='Model Weights')
    parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-batch-size', type=int, default=8, help='batch size for training [default: 8]')
    # Evaluation options
    parser.add_argument('-generate-place', type=str, default='./PON_Test/{}/{}'.format(Test_Path, steps))
    # parser.add_argument('-generate-place', type=str, default='./PON_Test/{}'.format(Test_Path))
    parser.add_argument('-Save-Features', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-Eval-CFP', action='store_true', default=False, help='enable the gpu')

    args = parser.parse_args()


    BACKBONE_DICT = {'IR-50': IR_50(112),
                 'Light_CNN_9': LightCNN_9Layers(),
                 'Light_CNN_29': LightCNN_29Layers(),
                 'Light_CNN_29_v2': LightCNN_29Layers_v2(),
                 'VGGFace2': resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth'),
                 'ArcFace_AVL_LP': ArcFace_AVL_LargePose([112, 112]),
                 'ArcFace_LP': ArcFace_LargePose([112, 112]),
    }
    Model = BACKBONE_DICT[args.model_select]
    print('Loading {}'.format(args.checkpoints))
    checkpoint = torch.load(args.checkpoints)
    Model.load_state_dict(checkpoint)

    save_dir = '{}'.format(args.generate_place)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
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
    transformed_dataset = FaceIdPoseDataset(args.csv_file, args.data_place,
                                            transform=transforms)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False)  # , num_workers=6)

    Features_List = {'Subject':[], 'Pose':[], 'ImgNum':[], 'Features':[]}
    count = 0
    minibatch_size = args.batch_size

    Frame = {}
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
                Frame.setdefault(ImgName, feature)
            count += minibatch_size
            print("Finish Processing {} images...".format(count))


    np.save('{}/Features.npy'.format(save_dir), Frame)

    # frequency = 2500  # Set Frequency To 2500 Hertz
    # duration = 1000  # Set Duration To 1000 ms == 1 second
    # winsound.Beep(frequency, duration)
