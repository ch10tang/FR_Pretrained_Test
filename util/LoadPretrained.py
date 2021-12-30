import torch
import torch.nn as nn


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module
    def forward(self, x):
        return self.module(x)


def LoadPretrained(Model, args):

    if args.model_select =='VGGFace2':
        print("Loading {} pretrained model".format(args.model_select))

    elif args.model_select == 'Light_CNN_9':
        print("Loading {} pretrained model".format(args.model_select))
        Model = WrappedModel(Model)
        checkpoint = torch.load('./Pretrained/LightCNN/LightCNN_9Layers_checkpoint.pth.tar')
        Model.load_state_dict(checkpoint['state_dict'])

    elif args.model_select == 'Light_CNN_29':
        print("Loading {} pretrained model".format(args.model_select))
        Model = WrappedModel(Model)
        checkpoint = torch.load('./Pretrained/LightCNN/LightCNN_29Layers_checkpoint.pth.tar')
        Model.load_state_dict(checkpoint['state_dict'])

    elif args.model_select == 'Light_CNN_29_v2':
        print("Loading {} pretrained model".format(args.model_select))
        Model = WrappedModel(Model)
        checkpoint = torch.load('./Pretrained/LightCNN/LightCNN_29Layers_V2_checkpoint.pth.tar')
        Model.load_state_dict(checkpoint['state_dict'])

    elif args.model_select == 'IR-50':
        print("Loading {} pretrained model".format(args.model_select))
        checkpoint = torch.load('./Pretrained/ms1m_ir50/backbone_ir50_ms1m_epoch63.pth')
        Model.load_state_dict(checkpoint)

    elif args.model_select == 'ArcFace_AVL_LP':
        print("Loading {} pretrained model".format(args.model_select))
        checkpoint = torch.load('./Pretrained/ms1m_ir50_AVLLargePose/Backbone_IR_50_Epoch_99_Batch_119493_Time_2020-09-27-10-18_checkpoint.pth')
        Model.load_state_dict(checkpoint)

    elif args.model_select == 'ArcFace_LP':
        print("Loading {} pretrained model".format(args.model_select))
        checkpoint = torch.load('./Pretrained/ms1m_ir50_LargePose/Backbone_ResNet_50_Epoch_40.pth')
        Model.load_state_dict(checkpoint)

    else:
        print('Please select valid pretrained model !')
        exit()

    print("Loading {} successfully!".format(args.model_select))

    return Model
