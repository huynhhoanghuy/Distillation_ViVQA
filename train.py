"""Main entrance for train/eval with/without KD on CIFAR-10"""

import argparse
import logging
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

import utils
import model.net as net
import model.data_loader as data_loader
import model.resnet as resnet
import model.wrn as wrn
import model.densenet as densenet
import model.resnext as resnext
import model.preresnet as preresnet
import model.base_model as base_model
import model.mobilenetv3 as mobilenetv3
from dataloaders.vivqa_dataset import ViVQADataset, VTCollator
from transformers import ViTFeatureExtractor, DetrFeatureExtractor, DeiTFeatureExtractor, \
                         AutoTokenizer, get_linear_schedule_with_warmup, \
                         YolosFeatureExtractor, YolosModel, YolosForObjectDetection
from evaluate import evaluate, evaluate_kd

parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir \
                    containing weights to reload before training")  # 'best' or 'train'




# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, metrics,val_dataloader, device, params,scheduler):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn_kd: 
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    ##################################################
    #############                           ##########
    #############                           ##########
    #############     TRAIN                 ##########
    #############                           ##########
    #############                           ##########
    ##################################################

    # set model to training mode
    model.train()
    # teacher_model.eval()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    count = 0   # count no. of data points
    output_batch_train = None
    output_batch_val = None
    # Use tqdm for progress bar
    with tqdm(dataloader, unit=' batch', total=len(dataloader)) as tq_data:
        for i, data in enumerate(tq_data):
            tq_data.set_description("Training ")

            # Every data instance is an input + img_label pair
            question, img, labels_batch = data['question'], data['image'], data['label']
            question, img, labels_batch = question.to(device), img.to(device), labels_batch.to(device)
            
            # one_hot_label = torch.nn.functional.one_hot(label, args.num_classes).float()
            print(question)
            print("-------------------")
            #count += labels_batch.size()[0]
            
            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            output_batch, student_feat_cnn, student_feat_vit = model.forward(img, question)
            output_batch = model.classify(output_batch)
            output_batch_train = output_batch


            model.eval()
            output_batch, student_feat_cnn, student_feat_vit = model.forward(img, question)
            output_batch = model.classify(output_batch)
            output_batch_val = output_batch
            print("output_batch_train:")
            print(output_batch_train)
            print("--------------")
            print("output_batch_val:")
            print(output_batch_val)
            print(torch.eq(output_batch_train,output_batch_val))
            exit()
            break



            loss = loss_fn_kd(output_batch, labels_batch)
            # print("loss:",loss)
            # clear previous gradients, compute gradients of all variables wrt loss
            # optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()
            

            # Evaluate summaries only once in a while
            # if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = F.softmax(output_batch, dim=1)
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                for metric in metrics}
            summary_batch['loss'] = loss.data
            summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data)

            tq_data.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            tq_data.update()
        scheduler.step()
    # print("!!!!!!!!!!!!!!!!!!!!!!")

    # print("summ[0]:",summ[0])
    # compute mean of all metrics in summary
    # metrics_mean = {}
    # for metric in summ[0]:
    #     temp = [x[metric].item() for x in summ]
    #     metrics_mean[metric] = np.mean(temp)
    # # metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- Train metrics: " + metrics_string)

    ##################################################
    #############                           ##########
    #############                           ##########
    #############     EVALUATE              ##########
    #############                           ##########
    #############                           ##########
    ##################################################


    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    # print("len(dataloader):",len(dataloader))
    # print(dataloader[0])
    # exit()
    # compute metrics over the dataset
    with tqdm(val_dataloader, unit=' batch', total=len(val_dataloader)) as tq_data:
        for i, data in enumerate(tq_data):
            # print("i: ",i)
            tq_data.set_description("Testing ")

            # Every data instance is an input + img_label pair
            question, img, labels_batch = data['question'], data['image'], data['label']
            question, img, labels_batch = question.to(device), img.to(device), labels_batch.to(device)
            
            # one_hot_label = torch.nn.functional.one_hot(label, args.num_classes).float()

            
            # Zero your gradients for every batch!

            # Make predictions for this batch
            output_batch, _, _ = model.forward(img, question)
            output_batch = model.classify(output_batch)

            output_batch_val = output_batch
            break

            # loss = loss_fn_kd(output_batch, labels_batch)
            loss = 0
            output_batch = F.softmax(output_batch, dim=1)
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            
            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                            for metric in metrics}
            # summary_batch['loss'] = loss.data[0]
            summary_batch['loss'] = loss
            summ.append(summary_batch)

    print("output_batch_train:")
    print(output_batch_train)
    print("--------------------")
    print("output_batch_val:")
    print(output_batch_val)
    print(torch.eq(output_batch_train,output_batch_val))
    exit()
    # compute mean of all metrics in summary

    metrics_mean = {}
    for metric in summ[0]:
        temp = [x[metric] for x in summ]
        metrics_mean[metric] = np.mean(temp)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return model


def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                       loss_fn_kd, metrics, params, model_dir, restore_file=None,scheduler=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) - file to restore (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    
    # Tensorboard logger setup
    # board_logger = utils.Board_Logger(os.path.join(model_dir, 'board_logs'))

    # learning rate schedulers for different models:
    # if params.model_version == "resnet18_distill" or params.model_version == "mobiNetv3_distill":
    #     scheduler = StepLR(optimizer, step_size=150, gamma=0.1)
    # # for cnn models, num_epoch is always < 100, so it's intentionally not using scheduler here
    # elif params.model_version == "cnn_distill": 
    #     scheduler = StepLR(optimizer, step_size=100, gamma=0.2) 

    for epoch in range(params.num_epochs):

        # scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        
        # compute number of batches in one epoch (one full pass over the training set)
        #val_metrics = evaluate_kd(model, val_dataloader, metrics, params, device, loss_fn_kd)

        model = train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader,
                 metrics, val_dataloader,device, params, scheduler)

        # Evaluate for one epoch on validation set
        # val_metrics = evaluate_kd(model, val_dataloader, metrics, params, device, loss_fn_kd)

        # val_acc = val_metrics['accuracy']
        # is_best = val_acc>=best_val_acc

        # # Save weights
        # utils.save_checkpoint({'epoch': epoch + 1,
        #                        'state_dict': model.state_dict(),
        #                        'optim_dict' : optimizer.state_dict()},
        #                        is_best=is_best,
        #                        checkpoint=model_dir)

        # # If best_eval, best_save_path
        # if is_best:
        #     logging.info("- Found new best accuracy")
        #     best_val_acc = val_acc
        #     # torch.save(model.state_dict(), "output/v1_" + str(epoch) + ".pt") 

        #     # Save best val metrics in a json file in the model directory
        #     best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
        #     utils.save_dict_to_json(val_metrics, best_json_path)

        # # Save latest val metrics in a json file in the model directory
        # last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        # utils.save_dict_to_json(val_metrics, last_json_path)




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
    args.data_dir = '/home/dmp/1.User/huy.hhoang/1/distill/ViVQA25'
    args.input = 'experiments/ViVQA/GuidedAtt_vit_resnet34_phobert_11_04_2023__11_31_10.pt'
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
    args.dropout = 0

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
    args.label_smooth = 0
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device
    return args

def replaceResNet(args, dataset, replacingModel):
    #Input of ResNet is 224x224x3
    #Output of ResNet is 7x7x512
    #I assumpt replacing model still have above in/out shape
    
    args.num_classes = dataset.num_classes                    
    model = base_model.build_GuidedAtt_replaceResNet(args, replacingModel)
    return model
    
def replaceResNetViT(args, dataset, replacingModelCNN, mobileNet):
    #Input of ResNet is 224x224x3
    #Output of ResNet is 7x7x512
    #I assumpt replacing model still have above in/out shape

    #Input of ViT is 224x224x3
    #Output of ViT is 197x768
    #I assumpt replacing model still have above in/out shape
    
    args.num_classes = dataset.num_classes                    
    model = base_model.build_GuidedAtt_replaceResNetViT(args, replacingModelCNN, mobileNet)
    return model
    


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()



    args_VQA = None
    torch.manual_seed(1234)
    torch.cuda.set_device('cuda:0')
    

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders, considering full-set vs. sub-set scenarios
    if params.subset_percent < 1.0:
        train_dl = data_loader.fetch_subset_dataloader('train', params)
    else:
        train_dl = data_loader.fetch_dataloader('train', params)
    
    dev_dl = data_loader.fetch_dataloader('dev', params)

    logging.info("- done.")

    """Based on the model_version, determine model/optimizer and KD training mode
       WideResNet and DenseNet were trained on multi-GPU; need to specify a dummy
       nn.DataParallel module to correctly load the model parameters
    """

    # train a 5-layer CNN or a 18-layer ResNet with knowledge distillation
    if params.model_version == "cnn_distill":
        model = net.Net(params).cuda() if params.cuda else net.Net(params)
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
        # fetch loss function and metrics definition in model files
        loss_fn_kd = net.loss_fn_kd
        metrics = net.metrics
    
    elif params.model_version == 'resnet18_distill':
        model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                momentum=0.9, weight_decay=5e-4)
        # fetch loss function and metrics definition in model files
        loss_fn_kd = net.loss_fn_kd
        metrics = resnet.metrics

    elif params.model_version == 'mobiNetv3_distill':
        #mobiNetCNN = mobilenetv3.mobilenetv3_small()
        #mobiNetViT = mobilenetv3.mobilenetv3_smallViT()
        args_VQA = get_arguments()
        dataset = ViVQADataset(args_VQA, pretrained=args_VQA.bert_pretrained, question_len=args_VQA.question_len,
                                mode='train', transform=None) #data_transforms[mode])for mode in ['train', 'test'] }

        #model = replaceResNetViT(args_VQA,dataset,mobiNetCNN,mobiNetViT)
        args_VQA.num_classes = dataset.num_classes

        model = base_model.build_GuidedAtt(args_VQA)
        # optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
        #                       momentum=0.9, weight_decay=5e-4)
        optimizer = optim.Adamax(filter(lambda x: x.requires_grad, model.parameters()), 
                            lr=args_VQA.init_lr, weight_decay=args_VQA.weight_decay)

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=args_VQA.warmup_steps,
                                                    num_training_steps=args_VQA.nepochs + 1,
                                                    last_epoch = -1)
        scheduler.step()
    
        if params.cuda:
            model = model.cuda()
        
        # fetch loss function and metrics definition in model files
        # loss_fn_kd = net.loss_fn_kd_intermediate
        loss_fn_kd = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=args_VQA.label_smooth)
        metrics = net.metrics
    """ 
        Specify the pre-trained teacher models for knowledge distillation
        Important note: wrn/densenet/resnext/preresnet were pre-trained models using multi-GPU,
        therefore need to call "nn.DaraParallel" to correctly load the model weights
        Trying to run on CPU will then trigger errors (too time-consuming anyway)!
    """
    if params.teacher == "resnet18":
        teacher_model = resnet.ResNet18()
        teacher_checkpoint = 'experiments/base_resnet18/best.pth.tar'
        teacher_model = teacher_model.cuda() if params.cuda else teacher_model

    elif params.teacher == "wrn":
        teacher_model = wrn.WideResNet(depth=28, num_classes=10, widen_factor=10,
                                        dropRate=0.3)
        teacher_checkpoint = 'experiments/base_wrn/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()

    elif params.teacher == "densenet":
        teacher_model = densenet.DenseNet(depth=100, growthRate=12)
        teacher_checkpoint = 'experiments/base_densenet/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()

    elif params.teacher == "resnext29":
        teacher_model = resnext.CifarResNeXt(cardinality=8, depth=29, num_classes=10)
        teacher_checkpoint = 'experiments/base_resnext29/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()

    elif params.teacher == "preresnet110":
        teacher_model = preresnet.PreResNet(depth=110, num_classes=10)
        teacher_checkpoint = 'experiments/base_preresnet110/best.pth.tar'
        teacher_model = nn.DataParallel(teacher_model).cuda()

    elif params.teacher == "ViVQA":
        args_VQA = get_arguments()
        device = torch.device(f"cuda:{args_VQA.gpu}" if torch.cuda.is_available() else "cpu")
        args_VQA.device = device
        torch.cuda.set_device(device)

        

        dataset = ViVQADataset(args_VQA, pretrained=args_VQA.bert_pretrained, question_len=args_VQA.question_len,
                                mode='test', transform=None) #data_transforms[mode])for mode in ['train', 'test'] }
        args_VQA.num_classes = dataset.num_classes
        tokenizer = AutoTokenizer.from_pretrained(args_VQA.bert_pretrained)
        ft_image_pretrained = args_VQA.vit_image_pretrained if args_VQA.model == 'GuidedAtt' else args_VQA.image_pretrained

        feature_extractor = ViTFeatureExtractor(do_resize=True, size=args_VQA.input_size, 
                                                    do_normalize=True, 
                                                    image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)
                                                ).from_pretrained(ft_image_pretrained)
        collator = VTCollator(feature_extractor, tokenizer, args_VQA.question_len)

        

        teacher_model = base_model.build_GuidedAtt(args_VQA)
        # teacher_checkpoint = 'experiments/ViVQA/GuidedAtt_vit_resnet34_phobert_11_04_2023__11_31_10.pt'
        # teacher_model.to(device)

        # model.load_state_dict(torch.load(args_VQA.input),strict=False)
        if args_VQA.input:
            print("LOADING pretrained teacher_model...")
            teacher_model.load_state_dict(torch.load(args_VQA.input))
            teacher_model.eval()

        teacher_model.to(device)
        layers = dict()
        def get_interlayer(name):
            def hook(model, input, output):
                layers[name] = output.detach()
            return hook
        teacher_model.visual_guided_atts[0].register_forward_hook(get_interlayer('a'))
        # subset = Subset(dataset, [58])  
        test_dl = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collator)

        train_dataset = ViVQADataset(args_VQA, pretrained=args_VQA.bert_pretrained, question_len=args_VQA.question_len,
                                mode='train', transform=None)
        # subset = Subset(dataset, [58])  
        train_dl = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=collator)                    

        args_VQA.num_classes = train_dataset.num_classes



    # utils.load_checkpoint(teacher_checkpoint, teacher_model)
    teacher_pytorch_total_params = sum(p.numel() for p in teacher_model.parameters())
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print("teacher_pytorch_total_params:",teacher_pytorch_total_params)
    # print("pytorch_total_params:",pytorch_total_params)
    # exit()
    # Train the model with KD
    logging.info("Experiment - model version: {}".format(params.model_version))
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    logging.info("First, loading the teacher model and computing its outputs...")


    train_and_evaluate_kd(model, teacher_model, test_dl, test_dl, optimizer, loss_fn_kd,
                            metrics, params, args.model_dir, args.restore_file,scheduler)

