import copy
import argparse
from datetime import datetime
import time
import os
from itertools import chain
import pandas as pd
import numpy as np

import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# from tensorboardX import SummaryWriter

# from model.ResNet import ResNet18, ResNet34, ResNet50, ResNet101 
# from model.segment_decoder import Decoder
import base_model

from dataloaders import custom_transforms as trforms
from dataloaders.vivqa_dataset import ViVQADataset, VTCollator
from transformers import ViTFeatureExtractor, DeiTFeatureExtractor, AutoTokenizer, get_linear_schedule_with_warmup
import utils


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')

    # Model setting
    
    parser.add_argument('--object_detection',  action='store_true', default=False, help='Use Object Detection model?')
    parser.add_argument('--vit_backbone', type=str, default='vit')
    parser.add_argument('--vit_image_pretrained', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--cnn_backbone', type=str, default='resnet34')
    parser.add_argument('--cnn_image_pretrained', type=str, default='google/vit-base-patch16-224-in21k')


    parser.add_argument('--backbone', type=str, default='vit')
    parser.add_argument('--bert_type', type=str, default='phobert')
    parser.add_argument('--bert_pretrained', type=str, default=None)
    parser.add_argument('--image_pretrained', type=str, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--data_dir', type=str, default='/content/dataset')
    parser.add_argument('--input', type=str, default='/content')
    parser.add_argument('--output', type=str, default='/content')
    
     # Define dimensions
    parser.add_argument('--v_vit_dim', type=int, default=768,
                        help='dim of image features')
    parser.add_argument('--v_cnn_dim', type=int, default=768,
                        help='dim of image features')

    parser.add_argument('--v_dim', type=int, default=768,
                        help='dim of image features')
    parser.add_argument('--q_dim', type=int, default=768,
                        help='dim of bert question features')
    parser.add_argument('--f_mid_dim', type=int, default=1024,
                        help='dim of middle layer of fusion layer')
    parser.add_argument('--joint_dim', type=int, default=512,
                        help='dim of joint features of fusion layer')
    
    # Multihead self-attention config
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help='dim of hidden layer of feed forward layers of transformers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of heads of transformers encoder')
    
    parser.add_argument('--glimpse', type=int, default=1,
                        help='number of glimpse for the attention reduction')

    # Choices of attention models
    parser.add_argument('--model', type=str, default='CMSA', choices=['CMSA', 'CrossAtt', 'GuidedAtt'],
                        help='the model we use')
    
    # Number of Co-Attention layers    
    parser.add_argument('--n_coatt', type=int, default=2,
                        help='dim of bert question features')   
    
    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')
    parser.add_argument('--question_len', default=20, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=None,
                        help='tfitrain_log_df word embedding?')
    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.2, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Training setting
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--resume_epoch', type=int, default=100)
    parser.add_argument('--train_fold', type=str, default='/content')
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--T', type=int, default=2)

    # Optimizer setting
    parser.add_argument('--init_lr', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--label_smooth', type=float, default=0.0)

    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=25)
    parser.add_argument('--emb_init', type=str, default='biowordvec', choices=['glove', 'biowordvec', 'biosentvec'])
    parser.add_argument('--self_att', action='store_true', default=False, help='Use Self Attention?')
    parser.add_argument('--use_spatial', action='store_true', default=False, help='Use spatial feature?')
    parser.add_argument('--use_cma', action='store_true', default=False, help='Use CMA?')
    parser.add_argument('--result_fold', type=str, default='results')

    return parser.parse_args()


def adjust_learning_rate(optimizer, lr_):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    torch.cuda.set_device(device)

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
    collator = VTCollator(feature_extractor, tokenizer, args.question_len)
    
    # Load train and validation dataset
    datasets = ViVQADataset(args, pretrained=args.bert_pretrained, question_len=args.question_len,
                                     mode='test', transform=None) #data_transforms[mode])for mode in ['train', 'test'] }
    
    dataloaders = DataLoader(datasets, batch_size=args.batch_size,
                                       shuffle=True, num_workers=2, collate_fn=collator)
    
    args.num_classes = datasets.num_classes
    
    print('Number of classes: ', args.num_classes)
    print("Dataset size: ", len(datasets))
    print("Dataloader size: ", len(dataloaders), len(dataloaders))
    
    # Create VQA model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(args)
    if args.input:
        print("LOADING pretrained model...")
        model.load_state_dict(torch.load(args.input))
        model.eval()

    model.to(device)
    
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    start_time = time.time()
    
    total_loss, total_acc = 0., 0.
    
    print('Start evaluating model...')
    # Start evaluate model:
    with torch.no_grad():

        # Loop over training data
        for i, data in enumerate(dataloaders):
            # Every data instance is an input + img_label pair
            question, img, label = data['question'], data['image'], data['label']
            question, img, label = question.to(device), img.to(device), label.to(device)
            # one_hot_label = torch.nn.functional.one_hot(label, args.num_classes).float()

            # Make predictions for this batch
            output = model.forward(img, question)
            # output = model.classify(output)

            # Compute the loss and accuracy
            loss = loss_fn(output, label)
            
            total_loss += loss.item()
            
            # Calculate accuracy for classification task
            acc = utils.calc_acc(output, label)
            total_acc += acc

    total_loss /= len(datasets)
    total_acc /= len(dataloaders)

    train_time = time.time() - start_time
    print('Validation complete in {:.0f}m {:.0f}s'.format(train_time // 60, train_time % 60))
    print('+ Validation accuracy: {:2f} %'.format(total_acc * 100))
    print('+ Validation loss: {:4f}'.format(total_loss))
    
    return model

if __name__ == '__main__':
    args = get_arguments()
    main(args)