from Model.model_irse import IR_50
from Model.model_lightcnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from Model.resnet50_ft_dims_2048 import resnet50_ft
import torch
import argparse
from util.LoadPretrained import LoadPretrained
from util.Validation_MPIE import Validation_MPIE
import os





if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Eval_SOTA_Model')
    # learning & saving parameters
    parser.add_argument('-data-place', type=str, default='F:/Database', help='prepared data path to run program')
    parser.add_argument('-csv-file', type=str, default='../DataList/MPIE_Warp7refs_GalleryProbe.csv', help='csv file to load image for training')
    parser.add_argument('-model-select', type=str, default='VGGFace2_TypeP_Sub4979_Img74522_60_92_WarpAffine7refs_Flip', help='Model Select')
    parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-batch-size', type=int, default=16, help='batch size for training [default: 8]')
    # Save options
    parser.add_argument('-save-place', type=str, default='./MultiPIE_Result')

    args = parser.parse_args()


    BACKBONE_DICT = {'IR-50': IR_50(112),
                 'Light_CNN_9': LightCNN_9Layers(),
                 'Light_CNN_29': LightCNN_29Layers(),
                 'Light_CNN_29_v2': LightCNN_29Layers_v2(),
                 'VGGFace2': resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth'),
                 'VGGFace2_TypeP_Sub4979_Img74522_60_92_WarpAffine7refs_Flip': IR_50(112)
    }
    try: BACKBONE = BACKBONE_DICT[args.model_select]
    except: BACKBONE = BACKBONE_DICT['IR-50'] # 假設都是IR-50 base model
    Model = LoadPretrained(BACKBONE, args)

    if args.cuda: device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Model = Model.to(device)
    Model.eval()

    save_dir = '{}/{}'.format(args.save_place, args.model_select)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    _ = Validation_MPIE(Model, 90, device, save_dir, args)
    print('Completed')
    exit()