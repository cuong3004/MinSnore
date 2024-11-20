from ghostpes.config import *
from cvnets.models.classification.mobilevit import MobileViT
from cvnets.modules import InvertedResidualSE
import torch 
import torch.nn as nn
# from ghostpes.ghostnet import ghostnet, module_ghost_1, module_ghost_2
from fvcore.nn import FlopCountAnalysis

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

## ghost 1
# model_ghost = ghostnet()

def get_mobile_vit(pretrained = False):
    model =  MobileViT(opts)
    if pretrained:
        model.load_state_dict(torch.load("mobilevit_xs.pt"))
    return model

# def get_ghost_vit_1(pretrained = False):
#     model_ghost_git_1 = get_mobile_vit(pretrained)

#     model_ghost_git_1.layer_1 = model_ghost.blocks[0]
#     return model_ghost_git_1

# ## ghost 2
# # model_ghost = ghostnet()
# def get_ghost_vit_2(pretrained = False):
#     model_ghost_git_2 = get_mobile_vit(pretrained)

#     model_ghost_git_2.layer_1 = model_ghost.blocks[0]
#     model_ghost_git_2.layer_2[0] = model_ghost.blocks[1]
#     return model_ghost_git_2

# ## ghost 3
# # model_ghost = ghostnet()

# def get_ghost_vit_3(pretrained = False):
#     model_ghost_git_3 = get_mobile_vit(pretrained)

#     model_ghost_git_3.layer_1 = model_ghost.blocks[0]
#     model_ghost_git_3.layer_2[0] = model_ghost.blocks[1]
#     model_ghost_git_3.layer_2[1] = model_ghost.blocks[2]
#     model_ghost_git_3.layer_2[2] = model_ghost.blocks[3]
#     return model_ghost_git_3


model_vit = get_mobile_vit()

import torch.nn as nn

class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

# module1 = InvertedResidualSE(opts, in_channels=16, out_channels=32, stride=1, expand_ratio=4, dilation=1, skip_connection=False)
# module2 = InvertedResidualSE(opts, in_channels=32, out_channels=48, stride=2, expand_ratio=4, dilation=1, skip_connection=False)
# module3 = InvertedResidualSE(opts, in_channels=48, out_channels=48, stride=1, expand_ratio=4, dilation=1, skip_connection=True)
# module4 = InvertedResidualSE(opts, in_channels=48, out_channels=48, stride=1, expand_ratio=4, dilation=1, skip_connection=True)
# module5 = InvertedResidualSE(opts, in_channels=48, out_channels=64, stride=2, expand_ratio=4, dilation=1, skip_connection=False)
# module6 = InvertedResidualSE(opts, in_channels=64, out_channels=80, stride=2, expand_ratio=4, dilation=1, skip_connection=False)
# module7 = InvertedResidualSE(opts, in_channels=80, out_channels=96, stride=2, expand_ratio=4, dilation=1, skip_connection=False)


# model_vit.layer_1[0] = module1
# model_vit.layer_2[0] = module2
# model_vit.layer_2[1] = IdentityLayer()
# model_vit.layer_2[2] = IdentityLayer()
# model_vit.layer_3[0] = module5
# model_vit.layer_4[0] = module6
# model_vit.layer_5[0] = module7

# a = model_vit.layer_1[0]
# b = model_vit.layer_2[0]
# c = model_vit.layer_2[1]
# d = model_vit.layer_2[2]
# e = model_vit.layer_3[0]
# f = model_vit.layer_4[0]
# g = model_vit.layer_5[0]


# print(f"{a}\n,{b}\n,{c}\n,{d}\n,{e}\n,{f}\n,{g}")

# model_vit = get_ghost_vit_1()

# layer_3_vit = model_vit.layer_3[1]

# print([i for i in layer_3_vit.state_dict()])

def count_parameters(model):
        total_trainable_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_trainable_params += params
        return total_trainable_params

input_shape = (1, 3, 128, 128)

total_params = count_parameters(model_vit)

flops = FlopCountAnalysis(model_vit, torch.ones((input_shape), dtype=torch.float32))
model_flops = flops.total()
print(f"Total Trainable Params: {round(total_params * 1e-6, 2)} M")
print(f"MAdds: {round(model_flops * 1e-6, 2)} M")
