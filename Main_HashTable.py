from Model.model_irse import IR_50
from Model.model_lightcnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from Model.resnet50_ft_dims_2048 import resnet50_ft
import torch
import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
from util.LoadPretrained import LoadPretrained
from util.DataLoader import FaceIdPoseDataset
from util.InputSize_Select import Transform_Select

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Eval_SOTA_Model')
    # learning & saving parameters
    parser.add_argument('-data-place', type=str, default='F:/Database/IJBC/IJBC_Official_ArcFace_112_112', help='prepared data path to run program')
    parser.add_argument('-csv-file', type=str, default='../DataList/IJBC_Official_Aligned.csv', help='csv file to load image for training')
    parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 8]')
    # Evaluation options
    parser.add_argument('-model-select', type=str, default='', help='Model Select')
    parser.add_argument('-model-path', type=str, default=None, help='The model path of the encoder')
    parser.add_argument('-epoch', type=int, default=None, help='The epoch of the encoder')
    parser.add_argument('-generate-place', type=str, default='./Test', help='prepared data path to run program')
    args = parser.parse_args()

    BACKBONE_DICT = {'IR-50': IR_50(112),
                 'Light_CNN_9': LightCNN_9Layers(),
                 'Light_CNN_29': LightCNN_29Layers(),
                 'Light_CNN_29_v2': LightCNN_29Layers_v2(),
                 # 'VGGFace2': resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth'),
    }
    if args.model_select in BACKBONE_DICT: BACKBONE = BACKBONE_DICT[args.model_select]
    else: BACKBONE = IR_50(112)
    Model = LoadPretrained(BACKBONE, args)

    save_dir = '{}/{}'.format(args.generate_place, args.model_select)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

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
    for i, batch_data in enumerate(dataloader):
   
        if args.model_select == 'VGGFace2': 
            #https://github.com/ox-vgg/vgg_face2/issues/17
            #batch_image = (batch_data[0]*255 - torch.Tensor([91.4953, 103.8827, 131.0912]).view((3,1,1))).to(device)
            batch_image = (batch_data[0]*255 - torch.Tensor([131.0912, 103.8827, 91.4953]).view((3, 1, 1))).to(device) #PIL default (RGB)
        else: batch_image = batch_data[0].to(device)

        _ = Model(batch_image)
        try: Model_Feature = Model.feature
        except: Model_Feature = Model.module.feature

        if len(Model_Feature.shape) > 2:
            Model_Feature = Model_Feature.view(Model_Feature.size(0), -1)
        features = (Model_Feature.data).cpu().numpy()
        batchImageName = batch_data[1]


        for feature, ImgName in zip(features, batchImageName):
            tmp = ImgName.split('/')
            Frame.setdefault('{}/{}'.format(tmp[-2], tmp[-1].split('.')[0]), feature)
        count += minibatch_size
        print("Finish Processing {} images...".format(count))

    
    np.save('{}/Features.npy'.format(save_dir), Frame)
