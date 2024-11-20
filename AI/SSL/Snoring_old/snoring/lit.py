# %%
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
import torch
import torch.nn as nn  
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torchmetrics.functional import accuracy, precision, recall, f1_score
from fvcore.nn import FlopCountAnalysis
average = 'macro'
from cvnets.layers.conv_layer import ConvLayer
from cvnets.layers.linear_layer import LinearLayer
import argparse
opts = argparse.Namespace()

setattr(opts,"model.classification.name","mobilevit_v3")
setattr(opts,"model.classification.classifier_dropout", 0.1)

setattr(opts,"model.classification.mit.mode" ,"xx_small_v3")
setattr(opts,"model.classification.mit.ffn_dropout", 0.0)
setattr(opts,"model.classification.mit.attn_dropout", 0.0)
setattr(opts,"model.classification.mit.dropout", 0.05)
setattr(opts,"model.classification.mit.number_heads", 4)
setattr(opts,"model.classification.mit.no_fuse_local_global_features", False)
setattr(opts,"model.classification.mit.conv_kernel_size", 3)

setattr(opts,"model.classification.activation.name", "swish")

setattr(opts,"model.normalization.name", "batch_norm_2d")
setattr(opts,"model.normalization.momentum", 0.1)

setattr(opts,"model.activation.name", "swish")

setattr(opts,"model.activation.layer.global_pool", "mean")
setattr(opts,"model.activation.layer.conv_init", "kaiming_normal")
setattr(opts,"model.activation.layer.linear_init", "trunc_normal")
setattr(opts,"model.activation.layer.linear_init_std_dev", 0.02)


def _get_dymn():
    from models.dymn.model import get_model as get_dymn
    model = get_dymn(pretrained_name="dymn04_as", width_mult=0.4, num_classes=2)
    return model

def _get_anntysnore_v1():
    from models.dymn.model import get_model as get_dymn
    model = get_dymn(pretrained_name="dymn04_as", width_mult=0.4, num_classes=2)
    print(model)
    model_vit = get_mobilevitv3()
    model_vit.layer_5
    
    print(model_vit.layer_4)
    
    print(model_vit.layer_3[1],)
    
    model.layers[4] = nn.Sequential(
        ConvLayer(opts, 16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
        model_vit.layer_3[1],
        ConvLayer(opts, 64, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
    model.layers[5] = torch.nn.Identity()


    model.layers[7] = nn.Sequential(
        ConvLayer(opts, 32, 80, kernel_size=(1, 1), stride=(1, 1), bias=False),
        model_vit.layer_4[1],
        ConvLayer(opts, 80, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
    model.layers[8] = torch.nn.Identity()
    model.layers[9] = torch.nn.Identity()
    model.layers[10] = torch.nn.Identity()
    model.layers[11] = torch.nn.Identity()
    # model.layers[12] = torch.nn.Identity()
    
    model.layers[13] = nn.Sequential(
        ConvLayer(opts, 64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
        model_vit.layer_5[1],
        ConvLayer(opts, 128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        )
    model.layers[14] = torch.nn.Identity()

    return model






def get_mobilevitv3():

    from cvnets.models.classification.mobilevit import MobileViTv3
    model =  MobileViTv3(opts)
    
    # state_dict = torch.load("checkpoint_ema_best.pt", map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    
    
    
    
    model.classifier.fc = LinearLayer(in_features=512, out_features=10, bias=True)
    model.conv_1 = ConvLayer(opts, 1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    return model

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(0,0), bias=False):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
    
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    F1_score = f1_score(pred_flat, labels_flat, average='macro')

    return accuracy_score(pred_flat, labels_flat), F1_score

class TFModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        orgin = 2
        self.f_module =  nn.Sequential(
            CNNBlock(1, 4, kernel_size=(1, 3), stride=(1, 2),  bias=False),
            CNNBlock(4, 8, kernel_size=(1, 3), stride=(1, 2), bias=False),
            CNNBlock(8, 16, kernel_size=(1, 3), stride=(1, 2), bias=False),
            CNNBlock(16, 16, kernel_size=(1, 3), stride=(1, 2), bias=False),
            nn.AdaptiveMaxPool2d((128, 1)),
            nn.Conv2d(16, 1, kernel_size=(1, 1), bias=False),
            nn.Sigmoid()
            # nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), bias=False),
            # nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), bias=False)
        )
        
        self.t_module =  nn.Sequential(
            CNNBlock(1, 4, kernel_size=(5, 1), stride=(2, 1), bias=False),
            CNNBlock(4, 8, kernel_size=(5, 1), stride=(2, 1), bias=False),
            CNNBlock(8, 16, kernel_size=(5, 1), stride=(2, 1), bias=False),
            CNNBlock(16, 16, kernel_size=(5, 1), stride=(2, 1), bias=False),
            nn.AdaptiveMaxPool2d((1, 345)),
            nn.Conv2d(16, 1, kernel_size=(1, 1), bias=False),
            nn.Sigmoid()
            # nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), bias=False),
            # nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 2), bias=False)
        )
    
    def forward(self, x):
        
        x_har = x[:,0:1]
        x_pess = x[:,1:2]
        
        # print(x_har.shape)
        # print("x_har", x_har.shape)
        # print("x_pess", x_pess.shape)
        
        f_score = self.f_module(x_har)
        # print("f_score", f_score.shape)
        
        # print("f_score", f_score.shape)
        
        t_score = self.t_module(x_pess)
        # print("t_score", t_score.shape)
        x_har = x_har + x_har * f_score 
        x_pess = x_pess + x_pess * t_score
        
        # print(x_har.shape)
        # x = x*torch.stack(f_score, t_score)
        
        
        
                
        return torch.concat((x_har, x_pess),1)

class LitClassification(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        from models.dymn.model import get_model as get_dymn
        # model = _get_anntysnore_v1()
        model = get_dymn(pretrained_name="dymn04_as", width_mult=0.4, num_classes=10)
        # model = get_mobilevitv3()
        self.model = model 
        
        # self.model = mobilenet_v3_small(pretrained=True)
        # self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 10)
        
        
        # self.model.features[0][0] = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        print(self.model)
        self.tf_module = TFModule()
        
        self.all_preds = []
        self.all_labels = []
    
    def forward(self, x):
        # print()
        x = x[:,:1]
        # print(x.shape)
        # x = self.tf_module(x)

        # print(x.shape)
        # x = self.model(x)[0]
        x = self.model(x)
        
        return x
        
        
        # forward
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)

        return optimizer
    
    def share_batch(self, batch, state):

        inputs, targets = batch['data'], batch['label']
        
        outputs = self(inputs)
        
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        if state == "Train":
            pass
            
            # acc_batch, f1_batch = flat_accuracy(outputs.cpu(), targets.cpu())
        
            # acc = self.acc(student_logits, labels)
            # self.log(f'{state}_acc', acc_batch, on_step=False, on_epoch=True, prog_bar=True)
        elif state == "valid":
            pred = outputs.argmax(dim=1)
            self.all_preds.append(pred.to('cpu'))
            self.all_labels.append(targets.to('cpu'))

        self.log(f"{state}_loss", loss, on_step=False, on_epoch=True)
#         self.log(f"{state}_rep_loss", rep_loss, on_step=False, on_epoch=True)
#         self.log(f"{state}_att_loss", att_loss, on_step=False, on_epoch=True)
#         self.log(f"{state}_cls_loss", cls_loss, on_step=False, on_epoch=True)

        return loss
    
    def on_validation_epoch_end(self):

        all_preds = torch.cat(self.all_preds,dim=0)
        all_labels = torch.cat(self.all_labels,dim=0)
        # print(all_preds.shape)
        acc = accuracy(all_preds, all_labels, task="multiclass", num_classes=10)
        pre = precision(all_preds, all_labels, task="multiclass", average=average, num_classes=10)
        rec = recall(all_preds, all_labels, task="multiclass", average=average, num_classes=10)
        f1 = f1_score(all_preds, all_labels, task="multiclass", average=average, num_classes=10)
        
        self.log('val_acc', acc,  prog_bar=True)
        self.log('val_pre', pre)
        self.log('val_rec', rec)
        self.log('val_f1', f1)
        
        self.all_preds = []
        self.all_labels = []
    
    def training_step(self, train_batch, batch_idx):
        loss = self.share_batch(train_batch, "train")
        # print(loss)
        return loss

    def validation_step(self, val_batch, batch_idx):

        loss = self.share_batch(val_batch, "valid")

    def test_step(self, test_batch, batch_idx):

        loss = self.share_batch(test_batch, "test")
        

        
# %%
if __name__ == "__main__":

    input_shape = (1, 1, 128, 128)

    x = torch.ones(input_shape) #87
    litmodel = LitClassification()  #TFModule()
    model = litmodel.model
    # model.conv_1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    # print(model)

    
    y = model(x)
    
    
    
    # y = litmodel(x)
    
    # print(y.shape)

    def count_parameters(model):
        total_trainable_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_trainable_params += params
        return total_trainable_params

    

    total_params = count_parameters(model)

    flops = FlopCountAnalysis(model, torch.ones((input_shape), dtype=torch.float32))
    model_flops = flops.total()
    print(f"Total Trainable Params: {round(total_params * 1e-6, 2)} M")
    print(f"MAdds: {round(model_flops * 1e-6, 2)} M")


    # print(get_mobilevitv3())
    # print(y[0].shape, y[1].shape)
    
# %%
# litmodel.model.conv_1 = Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, normalization=BatchNorm2d, activation='Swish', bias=False)


# # %%
# # from 

# # %%
# model = _get_dymn()
# # model_new = nn.Sequential(
# #     model.in_c,
# #     *model.layers[:7]
# # )

# model_new = nn.Sequential(
#     litmodel.model.conv_1,
#     litmodel.model.layer_1,
#     litmodel.model.layer_2,
#     litmodel.model.layer_3,
#     litmodel.model.layer_4,
#     litmodel.model.layer_5,
# )
# model_new(x[:,:1]).shape
# # %%
# # model.layers[:17]
# litmodel.model