import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models import ResNet50_Weights
#from torchvision.datasets import OxfordIIITPet
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.oxfordpets import  OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer_reference_impl import ImgSemSegTrainer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD
######################
from dlvc.models.segformer import  SegFormer


def train(args):
    #print("definisco trasformazioni")
    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])
    #print("trasformazioni definite")

    train_data = OxfordPetsCustom(root="oxford-iiit-pet", 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

    val_data = OxfordPetsCustom(root="oxford-iiit-pet", 
                            split="test",
                            target_types='segmentation', 
                            transform=val_transform,
                            target_transform=val_transform2,
                            download=True)
    #print("importazione effettuata")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = len(train_data.classes_seg)
    #SegFormer model
    model = SegFormer(num_classes=num_classes)
    
    model = DeepSegmenter(model)
    model.to(device)

    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Initialize metrics
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    
    trainer = ImgSemSegTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    args.num_epochs, 
                    model_save_dir,
                    batch_size=64,
                    val_frequency = val_frequency)
    trainer.train()

    trainer.save_miou_history()

    trainer.plot_miou_history(trainer.train_miou_history, trainer.val_miou_history)
    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    #trainer.dispose() 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-d', '--gpu_id', default='0', type=str, help='index of which GPU to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimization')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for regularization')
    parser.add_argument('--lr_step_size', type=int, default=5, help='Step size for learning rate scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma value for learning rate scheduler')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.gpu_id = 0
    args.num_epochs = 30
    ###############
    args.batch_size =  64                                           #  128
    args.learning_rate =  0.06                                 #0.001 #0.03         #0.06
    args.momentum =    0.9                                            #0.9
    args.weight_decay =    0.0001                                         #0.0001
    args.lr_step_size =    5                                          #5
    args.lr_gamma =     0.1                                             #0.1
    ################
    train(args)