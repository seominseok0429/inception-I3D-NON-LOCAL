import torch
from torch import nn

from i3d.i3dpt import I3D,Unit3Dpy
from i3d.i3dpt_non import I3D_NON, Unit3Dpy

def generate_model(opt):
    assert opt.model in ['i3d']
    if opt.model == 'i3d':
        if opt.dataset == 'hmdb51':
            # HMDB51 RGB
            model = I3D(num_classes=opt.n_classes, modality='rgb', dropout_prob=opt.dropout_prob)
            model_non = I3D_NON(num_classes=opt.n_classes, modality='rgb', dropout_prob=opt.dropout_prob)
            print('RGB')
        else:
            # HMDB51 FLOW
            model = I3D(num_classes=opt.n_classes, modality='flow', dropout_prob=opt.dropout_prob)
            model_non = I3D_NON(num_classes=opt.n_classes, modality='flow', dropout_prob=opt.dropout_prob)
            print('FLOW')        

    if not opt.no_cuda:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            print(opt.pretrain_path)
            model.load_state_dict(torch.load(opt.pretrain_path))
        if opt.model == 'i3d' and opt.non_local == False:
            model.conv3d_0c_1x1 = Unit3Dpy(
                in_channels=1024,
                out_channels=opt.n_finetune_classes,
                kernel_size=(1, 1, 1),
                activation=None,
                use_bias=True,
                use_bn=False)
            model = model.cuda()
            model = nn.DataParallel(model)
        elif opt.model == 'i3d' and opt.non_local == True:
            state_dict = model.state_dict()
            model_non.load_state_dict(state_dict, strict=False)
            model_non.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024, out_channels=51, 
                                               kernel_size=(1, 1, 1), activation=None, use_bias=True,
                                               use_bn=False)
            # batch norm frezze
            for i in model_non.children():
                for name1, j in i.named_children():
                    if name1 == 'batch3d':
                        for param in j.parameters():
                            param.requires_grad =False
                    if 'branch' in name1:
                        for name2, k in j.named_children():
                            if name2 == 'batch3d':
                                for param1 in k.parameters():
                                    param1.requires_grad =False
                            if name2 =='0' or name2=='1':
                                for name3, aa in k.named_children():
                                    if name3 == 'batch3d':
                                        for param2 in k.parameters():
                                            param2.requires_grad =False
            if opt.only_nonlocal:
                for child,b in model_non.named_children():
                    if child == 'NonLocalBlock_mixed_3' :
                        for param in b.parameters():
                            param.requires_grad = True
                    elif child == 'NonLocalBlock_mixed_4':
                        for param in b.parameters():
                            param.requires_grad = True
                    elif child == 'conv3d_0c_1x1':
                        for param in b.parameters():
                            param.requires_grad = True
                    else:
                        print(child)
                        for param in b.parameters():
                            param.requires_grad = False
            else:
                count = 0
                for child in model_non.children():
                    count += 1
                    if count < opt.ft_begin_index:
                        for param in child.parameters():
                            param.requires_grad = False
            model_non = model_non.cuda()
            model_non = nn.DataParallel(model_non)
            print('True')
            return model_non

        return model
