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

import torch
import torch.nn as nn
import torch.nn.functional as F


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
from transformers import ViTFeatureExtractor, DetrFeatureExtractor, DeiTFeatureExtractor, \
                         AutoTokenizer, get_linear_schedule_with_warmup, \
                         YolosFeatureExtractor, YolosModel, YolosForObjectDetection
import utils


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

class studentNet(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.
    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:
        Args:
            params: (Params) contains num_channels
        """
        super(studentNet, self).__init__()
        self.num_channels = params.num_channels
        
        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4*4*self.num_channels*4, self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
        self.fc2 = nn.Linear(self.num_channels*4, 10)       
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .
        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 32 x 32
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 32 x 32
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 16 x 16
        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 16 x 16
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 8 x 8
        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 8 x 8
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 4 x 4

        # flatten the output for each image
        s = s.view(-1, 4*4*self.num_channels*4)             # batch_size x 4*4*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
        s = self.fc2(s)                                     # batch_size x 10

        return s


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
    datasets = { mode: ViVQADataset(args, pretrained=args.bert_pretrained, question_len=args.question_len,
                                     mode=mode, transform=None) #data_transforms[mode])
                    for mode in ['train', 'test'] }
    
    dataloaders = { mode: DataLoader(datasets[mode], batch_size=args.batch_size,
                                     shuffle=True, num_workers=2, collate_fn=collator)
                    for mode in ['train', 'test'] }
    
    data_size =  { mode: len(datasets[mode]) for mode in ['train', 'test'] }
    
    args.num_classes = datasets['train'].num_classes
    
    print('Number of classes: ', args.num_classes)
    print("Dataset size: ", data_size)
    print("Dataloader size: ", len(dataloaders['train']), len(dataloaders['test']))
    
    # Create VQA model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(args)
    model.to(device)
    
    if args.print_summary:
        # dummy_image_feats = { 'pixel_values': torch.rand((args.batch_size, 3, args.input_size, args.input_size)).to(device) }
        # dummy_question = ['Hello, this is a question'] * args.batch_size
        # dummy_question_feats = tokenizer(dummy_question, padding='max_length', max_length=args.question_len, 
        #                                            truncation=True, return_tensors='pt').to(device)
        # print(summary(model, input_data=[dummy_image_feats, dummy_question_feats]), device=device)
        print(model)
        return model
    
    # Initialize optimizer algorithm
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.update_lr_every)
    optimizer = optim.Adamax(filter(lambda x: x.requires_grad, model.parameters()), 
                             lr=args.init_lr, weight_decay=args.weight_decay)

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.nepochs + 1,
                                                last_epoch = -1)
    scheduler.step()  # skip step with lr=0.0
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum', label_smoothing=args.label_smooth)
    
    EPOCHS = args.nepochs
    best_val_acc = 0.
    best_model = copy.deepcopy(model.state_dict())
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
                # model.train(phase == 'train')
                
                batch_loss, batch_acc = 0., 0.

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
                        output = model.forward(img, question)
                        output = model.classify(output)

                        # Compute the loss and accuracy
                        loss = loss_fn(output, label)
                        
                        batch_loss += loss.item()
                        
                        # Calculate accuracy for classification task
                        acc = utils.calc_acc(output, label)
                        batch_acc += acc

                        if phase == 'train':
                            # Backward model to compute its gradients
                            # optimizer.zero_grad()
                            loss.backward()

                            # Adjust learning weights
                            optimizer.step()
                        
                        tq_data.set_postfix_str(s='loss=%.4f, accuracy=%.4f' % (batch_loss / count, batch_acc / (i+1)))
                        tq_data.update()

                    if phase == 'train':
                        # adjust learning rate
                        scheduler.step()
                    
            batch_loss /= data_size[phase]
            batch_acc /= len(dataloaders[phase])
                        
            loggings = { **loggings,
                f'{phase}_loss': float(batch_loss),
                f'{phase}_acc': float(batch_acc),
            }
            
            # Save best model
            if phase == 'test' and batch_acc > best_val_acc:
                print('===> Saving this best model...')
                is_save_best_model = True
                best_val_acc = batch_acc
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, save_model_path_name) 
                # best_bert_model = copy.deepcopy(model.cmsa.q_emb.state_dict())
        
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
    
    # return model







#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------#


    import torch
    import torch.optim as optim
    from torchvision import datasets, transforms
    from KD_Lib.KD import VanillaKD

    # Define datasets, dataloaders, models and optimizers

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         "mnist_data",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #         ),
    #     ),
    #     batch_size=32,
    #     shuffle=True,
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         "mnist_data",
    #         train=False,
    #         transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #         ),
    #     ),
    #     batch_size=32,
    #     shuffle=True,
    # )

    teacher_model = model
    student_model = studentNet()

    teacher_optimizer = optim.SGD(teacher_model.parameters(), 0.01)
    student_optimizer = optim.SGD(student_model.parameters(), 0.01)

    # Now, this is where KD_Lib comes into the picture

    distiller = VanillaKD(teacher_model, student_model, train_loader, test_loader,
                        teacher_optimizer, student_optimizer)
    distiller.train_teacher(epochs=5, plot_losses=True, save_model=True)    # Train the teacher network
    distiller.train_student(epochs=5, plot_losses=True, save_model=True)    # Train the student network
    distiller.evaluate(teacher=False)                                       # Evaluate the student network
    distiller.get_parameters()                  



if __name__ == '__main__':
    args = get_arguments()
    main(args)

