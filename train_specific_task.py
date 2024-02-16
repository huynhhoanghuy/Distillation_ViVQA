from collections import OrderedDict

import copy
import argparse
from datetime import datetime
import time
import os
from itertools import chain
import pandas as pd
import numpy as np
from tqdm import tqdm
from torchinfo import summary
from classifier import SimpleClassifier, StudentSimpleClassifier
import torch
import torchvision.transforms as T

torch.cuda.empty_cache()

torch.cuda.memory_summary(device=None, abbreviated=False)

from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F
import mobilenetv3
# from tensorboardX import SummaryWriter

# from model.ResNet import ResNet18, ResNet34, ResNet50, ResNet101 
# from model.segment_decoder import Decoder
import base_model

from dataloaders import custom_transforms as trforms
from dataloaders.vivqa_dataset import ViVQADataset, VTCollator
from transformers import ViTFeatureExtractor, DetrFeatureExtractor, DeiTFeatureExtractor, \
                         AutoTokenizer, get_linear_schedule_with_warmup, \
                         YolosFeatureExtractor, YolosModel, YolosForObjectDetection
import utils

import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    
    # Choices of attention models
    parser.add_argument('--model', type=str, default='CrossAtt', choices=['CMSA', 'CrossAtt', 'GuidedAtt'],
                        help='the model we use')

    # Model setting
    parser.add_argument('--object_detection',  action='store_true', default=False, help='Use Object Detection model?')
    parser.add_argument('--vit_backbone', type=str, default='vit')
    parser.add_argument('--vit_image_pretrained', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--cnn_backbone', type=str, default='resnet34')
    parser.add_argument('--cnn_image_pretrained', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--bert_type', type=str, default='phobert')
    parser.add_argument('--bert_pretrained', type=str, default='vinai/phobert-base')
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--data_dir', type=str, default='/content/dataset')
    parser.add_argument('--output', type=str, default='/content')
    
    # Define dimensions
    parser.add_argument('--v_vit_dim', type=int, default=768,
                        help='dim of image features')
    parser.add_argument('--v_cnn_dim', type=int, default=768,
                        help='dim of image features')
    parser.add_argument('--q_dim', type=int, default=768,
                        help='dim of bert question features')
    parser.add_argument('--f_mid_dim', type=int, default=1024,
                        help='dim of middle layer of fusion layer')
    parser.add_argument('--joint_dim', type=int, default=512,
                        help='dim of joint features of fusion layer')
    parser.add_argument('--glimpse', type=int, default=1,
                        help='number of glimpse for the attention reduction')

    # Multihead self-attention config
    parser.add_argument('--hidden_dim', type=int, default=2048,
                        help='dim of hidden layer of feed forward layers of transformers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='number of heads of transformers encoder')
    
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
    parser.add_argument('--threshold', type=float, default=0.7)

    parser.add_argument('--print_summary', action='store_true', default=False, help='Print model summary?')

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

def replaceResNetViT(args, replacingModelCNN, mobileNet):
    #Input of ResNet is 224x224x3
    #Output of ResNet is 7x7x512
    #I assumpt replacing model still have above in/out shape

    #Input of ViT is 224x224x3
    #Output of ViT is 197x768
    #I assumpt replacing model still have above in/out shape
    
    model = base_model.build_GuidedAtt_replaceResNetViT(args, replacingModelCNN, mobileNet)
    return model


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.objects = 10

    torch.cuda.set_device(device)
    

    transform = transforms.RandomChoice([
                transforms.RandomRotation(15),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)) 
                ])
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained)
    if args.object_detection:
        feature_extractor = YolosFeatureExtractor(do_resize=True, size=args.input_size, 
                                                do_normalize=True, 
                                                image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)
                                                ).from_pretrained(args.vit_image_pretrained)
    else:
        feature_extractor = ViTFeatureExtractor(do_resize=True, size=args.input_size, 
                                                do_normalize=True, 
                                                image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)
                                                ).from_pretrained(args.vit_image_pretrained)
    collator = VTCollator(feature_extractor, tokenizer, args.question_len)
    
    # Load train and validation dataset
    datasets = {}
    datasets['train'] = ViVQADataset(args, pretrained=args.bert_pretrained, question_len=args.question_len,
                                     mode='train', transform=transform)
    datasets['test'] = ViVQADataset(args, pretrained=args.bert_pretrained, question_len=args.question_len,
                                     mode='test', transform=None)

    dataloaders = {}
    dataloaders['train'] = DataLoader(datasets['train'], batch_size=args.batch_size,
                                     shuffle=True, num_workers=1, collate_fn=collator)
    dataloaders['test'] = DataLoader(datasets['test'], batch_size=1,
                                     shuffle=False, num_workers=1, collate_fn=collator)

    data_size =  { mode: len(datasets[mode]) for mode in ['train', 'test'] }
    
    args.num_classes = datasets['train'].num_classes
    
    print('Number of classes: ', args.num_classes)
    print("Dataset size: ", data_size)
    print("Dataloader size: ", len(dataloaders['train']), len(dataloaders['test']))
    
    # Create VQA model
    constructor = 'build_%s' % args.model
    mobiNetCNN = mobilenetv3.mobilenetv3_small()
    mobiNetViT = mobilenetv3.mobilenetv3_smallViT()

    current_model_dict = mobiNetCNN.state_dict()
    loaded_state_dict = torch.load("./pretrained_mobileNet/mobilenetv3-small-55df8e1f.pth")
    new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
    mobiNetCNN.load_state_dict(new_state_dict, strict=False)

    current_model_dict = mobiNetViT.state_dict()
    loaded_state_dict = torch.load("./pretrained_mobileNet/mobilenetv3-small-55df8e1f.pth")
    new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
    mobiNetViT.load_state_dict(new_state_dict, strict=False)
    args.num_classes = 353
    m_base_model = replaceResNetViT(args, mobiNetCNN,mobiNetViT)
 
    m_base_model.load_state_dict(torch.load("../GuidedAtt_vit_resnet34_phobert_31_12_2023__01_19_41.pt"), strict=True)
    for param in m_base_model.parameters():
        param.requires_grad = False
    
    m_base_model.to(device)
    m_base_model.eval()
    args.num_classes = 10
    args.activation = 'leakyrelu'
    student_model = StudentSimpleClassifier(args.joint_dim, args.joint_dim*2 , args.num_classes, args)

    # student_model.load_state_dict(torch.load("/home/dmp/1.Users/05.huy.hhoang/1/GuidedAtt_vit_resnet34_phobert_19_07_2023__16_30_34_bk.pt"),strict=True)
    student_pytorch_total_params = sum(p.numel() for p in student_model.parameters())
    pytorch_total_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    print("student_pytorch_total_params:",student_pytorch_total_params)

    base_pytorch_total_params = sum(p.numel() for p in m_base_model.parameters())
    pytorch_total_params = sum(p.numel() for p in m_base_model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    print("base_pytorch_total_params:",base_pytorch_total_params)
    
    student_model.to(device)
    pretrained_dict = None
    del pretrained_dict

    


    # Initialize optimizer algorithm
    # momentum = 0.8
    optimizer = optim.SGD(filter(lambda x: x.requires_grad, student_model.parameters()), lr=args.init_lr, momentum=0.9, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(filter(lambda x: x.requires_grad, student_model.parameters()), lr=args.init_lr, weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(filter(lambda x: x.requires_grad, student_model.parameters()), lr=args.init_lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.update_lr_every)
    # optimizer = optim.Adamax(filter(lambda x: x.requires_grad, student_model.parameters()), 
    #                          lr=args.init_lr, weight_decay=args.weight_decay)
    # optimizer = optim.RMSprop(filter(lambda x: x.requires_grad, student_model.parameters()), lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.nepochs + 1,
                                                last_epoch = -1)
    scheduler.step()  # skip step with lr=0.0
    # Initialize loss function
    loss_CE = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=args.label_smooth)


    loss_KD = torch.nn.KLDivLoss()

    def earth_mover_distance(y_true, y_pred, bias = 0.25):
        y_one_hot = F.one_hot(y_true, 10)
        new_y_true = F.softmax(y_one_hot.to(torch.double),dim=-1)
        new_y_pred = F.softmax(y_pred, dim=-1)
        # EMD = torch.mean(torch.sum(torch.square(torch.cumsum(new_y_true, dim=-1) - torch.cumsum(new_y_pred, dim=-1)), dim=-1))
        EMD = torch.square(torch.cumsum(new_y_true, dim=-1) - torch.cumsum(new_y_pred, dim=-1))
        return torch.mean(torch.sum((new_y_pred)*(EMD + bias),dim = -1))
    
    

    def loss_fn_kd_intermediate(outputs, labels, teacher_outputs, student_vit, teacher_vit, student_cnn, teacher_cnn):
       
        alpha = 0.9
        T = 5

        total_loss = (1-alpha)* loss_CE(outputs,labels)  + \
                    (alpha*1/3)* T * T* loss_CE(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) +\
                     (alpha*1/3)* loss_KD(F.log_softmax(student_vit, dim=1), F.softmax(teacher_vit, dim=1)) +\
                     (alpha*1/3)* loss_KD(F.log_softmax(student_cnn, dim=1), F.softmax(teacher_cnn, dim=1))

                

        return total_loss
    
    EPOCHS = args.nepochs
    best_val_acc = 0.
    print("------------------------------------------")
    best_model = copy.deepcopy(student_model.state_dict())
    # best_bert_model = copy.deepcopy(model.q_embedding.state_dict())
    
    start_train_time = time.time()
    
    # save best model weights
    save_dir = args.output
    now = datetime.now()
    now_str = now.strftime("%d_%m_%Y__%H_%M_%S")
    best_model_filename = '{}_{}_{}_{}_{}.pt'.format(args.model, args.vit_backbone, args.cnn_backbone, args.bert_type, now_str)
    save_model_path_name = os.path.join(save_dir, best_model_filename)
    # biobert_path_name = os.path.join(save_dir, '{}_{}_{}.pt'.format(args.bert_type, args.backbone, now_str))
    
    # save train log
    train_log_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 
                                        'val_loss''val_acc', 'is_best_model'])
    csv_path = os.path.join(save_dir, f'train_log_{best_model_filename}.csv')
    loss_MSE = torch.nn.MSELoss()
    
    count_0_99 = 0
    emd_weight = 0
    for epoch in range(EPOCHS):
        print(f'\nEPOCH {epoch}/{EPOCHS - 1}')
        print('lr = ', scheduler.get_last_lr())
        print('-' * 10)
        start_epoch_time = time.time()
        
        loggings = {
            'epoch': epoch
        }
        
        is_save_best_model = False
        
        for phase in ['train', 'test']:
    
            with torch.set_grad_enabled(phase == 'train'):        
                # Make sure gradient tracking is on if in training, and do a pass over the data
                student_model.train(phase == 'train')
                batch_loss, batch_acc = [], []
                count = 0   # count no. of data points
                # Loop over training data
                with tqdm(dataloaders[phase], unit=' batch', total=len(dataloaders[phase])) as tq_data:
                    for i, data in enumerate(tq_data):
                        
                        tq_data.set_description("Training " if phase == 'train' else "Validation ")
                        
                        # Every data instance is an input + img_label pair
                        question, img, label = data['question'], data['image'], data['label']

                        question, img, label = question.to(device), img.to(device), label.to(device)
                        # one_hot_label = torch.nn.functional.one_hot(label, args.num_classes).float()

                        count += label.size()[0]
                        
                        # Zero your gradients for every batch!
                        optimizer.zero_grad()

                        # Make predictions for this batch
                        raw_output, v_feat_vit, v_feat_cnn = m_base_model.forward(img, question)
                        raw_output = raw_output.detach()

                        output = student_model.forward(raw_output)


                        if phase == 'train':                            
                            if epoch ==args.warmup_steps:
                                emd_weight = (loss_CE(output, label) / earth_mover_distance(label,output))
                                emd_weight = emd_weight.detach()
                            elif epoch ==(args.warmup_steps*2):
                                emd_weight = (loss_CE(output, label) / earth_mover_distance(label,output)) / 2
                                emd_weight = emd_weight.detach()
                            # loss = emd_weight*earth_mover_distance(label,output) + loss_MSE(output,new_y_true.to(torch.float))
                            loss = emd_weight*earth_mover_distance(label,output) + loss_CE(output, label)
                            
                        else:
                            loss = emd_weight*earth_mover_distance(label,output)+ loss_CE(output, label)

                        
                        batch_loss.append(loss.item())
                        
                        # Calculate accuracy for classification task
                        acc = utils.calc_acc(output, label)
                        batch_acc.append(acc)
                        if phase == 'train':
                            # Backward model to compute its gradients
                            # optimizer.zero_grad()
                            loss.backward()

                            # Adjust learning weights
                            optimizer.step()
                        
                        tq_data.set_postfix_str(s='loss=%.4f, accuracy=%.4f' % (sum(batch_loss) / len(batch_loss), sum(batch_acc) / len(batch_acc)))
                        tq_data.update()

                    if phase == 'train':
                        # adjust learning rate
                        scheduler.step()

            loggings = { **loggings,
                f'{phase}_loss': float(sum(batch_loss) / len(batch_loss)),
                f'{phase}_acc': float(sum(batch_acc) / len(batch_acc)),
            }
            
            # Save best model
            if phase == 'test' and (sum(batch_acc) / len(batch_acc)) > best_val_acc:
                print('===> Saving this best model...',save_model_path_name)
                is_save_best_model = True
                best_val_acc = sum(batch_acc) / len(batch_acc)
                best_model = copy.deepcopy(student_model.state_dict())
                torch.save(best_model, save_model_path_name) 
                # best_bert_model = copy.deepcopy(model.cmsa.q_emb.state_dict())
            if sum(batch_acc) / len(batch_acc) > 0.99:
                count_0_99 += 1
                if count_0_99 > 8:
                    exit()
        row = pd.Series(data={
            **loggings,
            'is_best_model': int(is_save_best_model)
        })
        train_log_df = train_log_df.append(row, ignore_index=True)
        # save train log:
        train_log_df.to_csv(csv_path, index=False, mode='w+')  # overwrite mode

        time_epoch_elapsed = time.time() - start_epoch_time
        print('Epoch time: {:.0f}m {:.0f}s'.format(time_epoch_elapsed // 60, time_epoch_elapsed % 60))


    train_time = time.time() - start_train_time
    print('Training complete in {:.0f}m {:.0f}s'.format(train_time // 60, train_time % 60))
    print('Best val accuracy: {:4f}'.format(best_val_acc))
    print('Best model save to: ', best_model_filename)
    # torch.save(best_bert_model, biobert_path_name)
    
    return student_model

if __name__ == '__main__':
    # mem = get_gpu_memory()
    # while(mem[0] <= 35000):
    #     mem = get_gpu_memory() # total free-memory
    args = get_arguments()
    main(args)
