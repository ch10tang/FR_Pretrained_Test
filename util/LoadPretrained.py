import torch
import torch.nn as nn
import os

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

    else:
        epoch = 90 if args.epoch is None else args.epoch
        model_path = './Pretrained' if args.model_path is None else args.model_path
        if os.path.exists('{}/{}/Backbone_IR_50_Epoch_{}.pth'.format(model_path, args.model_select, epoch)):
            checkpoint = torch.load('{}/{}/Backbone_IR_50_Epoch_{}.pth'.format(model_path, args.model_select, epoch))
            Model.load_state_dict(checkpoint)
            args.model_select = model_path.split('/')[-1]
        else:
            print('Please select valid pre-trained model!')
            exit()
    print("Loading {} successfully!".format(args.model_select))

    return Model
