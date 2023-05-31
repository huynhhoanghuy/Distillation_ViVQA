import copy
import argparse
from datetime import datetime
import time
import os
from itertools import chain
import matplotlib
import pandas as pd
import numpy as np

import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from torch.nn import functional as F
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

# from tensorboardX import SummaryWriter

# from model.ResNet import ResNet18, ResNet34, ResNet50, ResNet101 
# from model.segment_decoder import Decoder
import base_model

from dataloaders import custom_transforms as trforms
from dataloaders.vivqa_dataset import ViVQADataset, VTCollator
from transformers import ViTFeatureExtractor, DeiTFeatureExtractor, AutoTokenizer, get_linear_schedule_with_warmup
import utils

def get_arguments():
    # Model setting
    
    args = argparse.ArgumentParser()
    args.gpu = 0

    args.object_detection = False
    args.vit_backbone = 'vit'
    args.vit_image_pretrained ='google/vit-base-patch16-224-in21k'
    args.cnn_backbone='resnet34'
    args.cnn_image_pretrained='resnet34'

    args.indices = [1, 2, 3]
    
    args.bert_type = 'phobert'
    args.bert_pretrained = 'vinai/phobert-base'

    args.input_size = 224
    args.data_dir = '/data/huy.hhoang/ViVQA25'
    args.input = '/data/huy.hhoang/ViVQA_model/output/trained_models/GuidedAtt/GuidedAtt_vit_resnet34_phobert_11_04_2023__11_31_10.pt'
    args.output  = ''
    
     # Define dimensions
    args.v_vit_dim = 768
    args.v_cnn_dim = 512

    args.q_dim = 768
    args.f_mid_dim = 1024
    args.joint_dim = 1024
    
    # Multihead self-attention config
    args.hidden_dim = 2048
    args.num_heads = 8
    
    args.glimpse = 1

    # Choices of attention models
    args.model = 'GuidedAtt'
    
    args.question_len = 20
    
    # Activation function + dropout for classification module
    args.activation = 'relu'
    args.dropout = 0.3

    # Training setting
    args.seed = 1234
    args.batch_size = 1
    args.nepochs = 40

    # Optimizer setting
    args.init_lr = 1e-4
    args.max_lr = 5e-5
    args.weight_decay = 1e-5
    args.momentum=0.9
    args.warmup_steps = 5
    args.label_smooth = 0.

    return args


def adjust_learning_rate(optimizer, lr_):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

args = get_arguments()

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

torch.cuda.set_device(device)

# Load train and validation dataset
dataset = ViVQADataset(args, pretrained=args.bert_pretrained, question_len=args.question_len,
                                    mode='test', transform=None) #data_transforms[mode])for mode in ['train', 'test'] }


# data_transforms = {
#     'train': transforms.Compose([
#         trforms.FixedResize(size=(args.input_size, args.input_size)),
#         trforms.RandomHorizontalFlip(),
#         trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         trforms.ToTensor()
#     ]),
#     'test': transforms.Compose([
#         trforms.FixedResize(size=(args.input_size, args.input_size)),
#         trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         trforms.ToTensor()
#     ]),
# }
tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained)
ft_image_pretrained = args.vit_image_pretrained if args.model == 'GuidedAtt' else args.image_pretrained
feature_extractor = ViTFeatureExtractor(do_resize=True, size=args.input_size, 
                                            do_normalize=True, 
                                            image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)
                                        ).from_pretrained(ft_image_pretrained)
collator = VTCollator(feature_extractor, tokenizer, args.question_len, store_origin_data=True)

args.num_classes = dataset.num_classes
# Create VQA model
constructor = 'build_%s' % args.model
model = getattr(base_model, constructor)(args)
if args.input:
    print("LOADING pretrained model...")
    model.load_state_dict(torch.load(args.input))
    model.eval()

model.to(device)

layers = dict()
def get_interlayer(name):
    def hook(model, input, output):
        layers[name] = output.detach()
    return hook

model.visual_guided_atts[0].register_forward_hook(get_interlayer('a'))


subset = Subset(dataset, [58])  
# subset = Subset(dataset, [52, 58, 68, 77])  

dataloader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collator)

args.num_classes = dataset.num_classes

print('Number of classes: ', args.num_classes)
print("Dataset size: ", len(dataset))
print("Dataloader size: ", len(dataloader), len(dataloader))
print('Start evaluating model...')

# Initialize loss function
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
start_time = time.time()
total_loss, total_acc = 0., 0.

def show_img_overlay(img1, overlay, alpha=0.8):
    img1 = np.asarray(img1)
    img2 = np.asarray(overlay)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img1)
    # plt.imshow(img2, alpha=alpha, cmap='rainbow')
    plt.axis("off")
    plt.savefig("img1.png")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img2, alpha=alpha, cmap='rainbow')
    plt.axis("off")
    plt.savefig("overlay.png")
    plt.close(fig)

# Start evaluate model:
with torch.no_grad():

    # Loop over training data
    for i, data in enumerate(dataloader):
        # Every data instance is an input + img_label pair
        print('\n--------')
        print(f'Inference {i+1}:')
        org_question, org_img, answer = data['org_question'], data['org_image'][0], data['answer'][0]
        question, img, label = data['question'], data['image'], data['label']
        question, img, label = question.to(device), img.to(device), label.to(device)
        # one_hot_label = torch.nn.functional.one_hot(label, args.num_classes).float()
        
        print('Image: ')
        # plt.imshow(transforms.ToPILImage()(transforms.ToTensor()(org_img)), interpolation="bicubic")
        # plt.show()

        print('Question: ', org_question[0])
        print(f'Answer: {answer}; \tLabel index: {label[0]}')

        # Make predictions for this batch
        output = model.forward(img, question)
        # output = model.classify(output)
        
        print(layers['a'].shape)
        attn_map = layers['a'].mean(dim=2)[0, 1:].cpu()
        attn_map = F.softmax(attn_map)
        attn_map = F.interpolate(attn_map.view(1, 1, 14, 14), (224, 224), mode='bicubic').view(224, 224, 1)

        print(attn_map.shape)
        show_img_overlay(org_img, attn_map, alpha=0.2)
        # Compute the loss and accuracy
        loss = loss_fn(output, label)
        
        total_loss += loss.item()
        
        # Calculate accuracy for classification task
        acc = utils.calc_acc(output, label)
        total_acc += acc
                    
        y_pred = torch.nn.functional.softmax(output, dim=1)
        y_pred = torch.argmax(y_pred, dim=1, keepdim=False)
        
        label_pred = int(y_pred[0])
        print('Model prediction: Answer: {}; Label indice: {}'.format(dataset.id2ans[label_pred], label_pred))


total_loss /= len(dataset)
total_acc /= len(dataloader)

train_time = time.time() - start_time
print('Inference complete in {:.0f}m {:.0f}s'.format(train_time // 60, train_time % 60))
print('+ Inference accuracy: {:2f} %'.format(total_acc * 100))
print('+ Inference loss: {:4f}'.format(total_loss))