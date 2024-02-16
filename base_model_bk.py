"""
This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""

import torch
import torch.nn as nn
from attention import BiAttention, StackedAttention
from co_attention import CoTransformerBlock, FusionAttentionFeature, GuidedTransformerEncoder, AttentionReduce, FusionLinear
from language_model import WordEmbedding, QuestionEmbedding, BertQuestionEmbedding, SelfAttention
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
# from utils import tfidf_loading, generate_spatial_batch
from simple_cnn import SimpleCNN
from auto_encoder import Auto_Encoder_Model
from backbone import initialize_backbone_model, ObjectDetectionModel
# from multi_task import ResNet50, ResNet18, ResNet34
from mc import MCNet
from convert import Convert, GAPConvert
import os
from non_local import NONLocalBlock3D
from transformer.SubLayers import MultiHeadAttention
import utils
from transformers import ViTFeatureExtractor, DetrFeatureExtractor, DeiTFeatureExtractor, \
                         AutoTokenizer, get_linear_schedule_with_warmup, \
                         YolosFeatureExtractor, YolosModel, YolosForObjectDetection, AutoImageProcessor

from MoE import GatingFunction



class CrossAttentionModel(nn.Module):

    def __init__(self, q_emb, v_emb, co_att_layers, fusion, classifier, args) -> None:
        super(CrossAttentionModel, self).__init__()
        self.q_emb = q_emb
        self.v_emb = v_emb
        self.classifier = classifier
        self.co_att_layers = co_att_layers
        self.fusion = fusion
        self.flatten = nn.Flatten()
        self.args = args
        
    def forward(self, v, q):
        v_emb = self.v_emb(v)
        q_emb = self.q_emb(q)
        
        # q_emb = q_emb[:, 0, :]
        # v_emb = v_emb[:, 0, :]
        
        # q_emb = q_emb.mean(1, keepdim =True)
        # v_emb = v_emb.mean(1, keepdim =True)
        # v_emb = v_emb.repeat_interleave(self.args.question_len, 1)
        
        for co_att_layer in self.co_att_layers:
            v_emb, q_emb = co_att_layer(v_emb, q_emb)
        
        if self.fusion:
            out = self.fusion(v_emb, q_emb)
        else:
            v_emb = v_emb.mean(1, keepdim =True)
            v_emb = v_emb.repeat_interleave(self.args.question_len, 1)
            
            out = q_emb * v_emb
        
        out = out.mean(1, keepdim =True)
        out = self.flatten(out)
        
        # out = out.permute((0, 2, 1))
        # out = out.mean(dim=-1)
        
        return out
    
    def classify(self, x):
        return self.classifier(x)
    

def build_CrossAtt(args):
    print('Use BERT as question embedding...')
    q_dim = args.q_dim
    q_emb = BertQuestionEmbedding(args.bert_pretrained, args.device, use_mhsa=True)
    utils.set_parameters_requires_grad(q_emb, True)

    print('Loading image feature extractor...')
    v_dim = args.v_dim
    if args.object_detection:
        v_emb = ObjectDetectionModel(args.image_pretrained, args.threshold, args.question_len)
        utils.set_parameters_requires_grad(v_emb, False)  # freeze Object Detection model
    else:
        v_emb = initialize_backbone_model(args.backbone, use_imagenet_pretrained=args.image_pretrained)[0]

    coatt_layers = nn.ModuleList([])
    for _ in range(args.n_coatt):
        coatt_layers.append(CoTransformerBlock(v_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout))

    fusion = None
    if args.object_detection:
        fusion = FusionAttentionFeature(args)

    classifier = SimpleClassifier(
        args.joint_dim, args.joint_dim * 2, args.num_classes, args)

    return CrossAttentionModel(q_emb, v_emb, coatt_layers, fusion, classifier, args)

class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus ReLU'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.ReLU(x + y)

def tile_2d_over_nd(feature_vector, feature_map):
    """ Repeat the same feature vector over all spatial positions of a given feature map.
        The feature vector should have the same batch size and number of features as the feature map.
    """
    # print(feature_vector.size())
    n, _, c = feature_vector.size()
    spatial_sizes = feature_map.size()

    tiled = feature_vector.view(n,  *([1] * len(spatial_sizes)), c)

    tiled = feature_vector.view(n,  *([1] * len(spatial_sizes)), c).expand(n, c, *spatial_sizes)
    return tiled

class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.ReLU = nn.ReLU(inplace=True)
        self.fusion = Fusion()

    def forward(self, v, q):
        q_in = q
        v = self.v_conv(self.drop(v))
        q = self.q_lin(self.drop(q))

        q = tile_2d_over_nd(q, v)

        x = self.fusion(v, q)
        x = self.x_conv(self.drop(x))
        return x


class GuidedAttentionModel(nn.Module):
    def __init__(self, q_emb, v_embs, visual_guided_atts, visual_reduces, fusion, q_guided_att, classifier, od_visual_vit_guided_att,
                od_question_guided_att, od_fusion, args):
        super(GuidedAttentionModel, self).__init__()
        self.objects = 9
        self.q_emb = q_emb
        
        self.v_embs = nn.ModuleList(v_embs)
        self.visual_guided_atts = nn.ModuleList(visual_guided_atts)
        self.od_visual_vit_guided_att = od_visual_vit_guided_att
        self.visual_reduces = nn.ModuleList(visual_reduces)
        
        self.fusion = fusion
        self.od_fusion = od_fusion
        self.q_guided_att = q_guided_att
        self.od_question_guided_att = od_question_guided_att
        self.question_reduced = AttentionReduce(768, 768 // 2, 1)

        self.classifier = classifier
        self.flatten = nn.Flatten()
        self.counter = Counter(self.objects)
        # self.visual_bbox_guided_att = od_visual_vit_guided_att
        self.gating_function = GatingFunction(input_size = 768*20, hidden_size = 256, num_experts = 2)
        self.num_classes =  args.num_classes
        self.gpu = args.gpu
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        # self.ID_number = {
        #     1: 323,
        #     2: 329,
        #     3: 26,
        #     4: 46,
        #     5: 106,
        #     6: 214,
        #     7: 312,
        #     8: 209,
        #     9: 201,
        #     10: 263
        # }
        # 1 object -> "một" -> id in dataset = 323

        # self.zero_classes_onehot = torch.zeros((args.batch_size, self.num_classes))
        # self.zero_classes_onehot = self.zero_classes_onehot.to("cuda:0")
    
    def forward(self, v, q, b):
        q_feat = self.q_emb(q)
        gate_output = self.gating_function(q_feat)

        od_v_guided = self.od_visual_vit_guided_att(b, q_feat)   
        od_q_guided = self.od_question_guided_att(q_feat, b)     
        a = self.od_fusion(od_q_guided.mean(1), od_v_guided.mean(1))
        one_hot_conf = self.counter(b, a)
        

        
        ############# map to classes of VQA dataset #################

        zero_classes_onehot = torch.zeros((q_feat.shape[0],self.num_classes))
        # print("zero_classes_onehot:",zero_classes_onehot.shape)
        # print("one_hot_conf:",one_hot_conf.shape)
        zero_classes_onehot = zero_classes_onehot.to(self.device)
        for batch in range(zero_classes_onehot.shape[0]):
            for i in range(10):
                one_hot_conf[batch, i]
                self.ID_number[i+1]
                zero_classes_onehot[batch, self.ID_number[i+1]] = one_hot_conf[batch, i]

        ############# map to classes of VQA dataset #################


        v_feats = []
        v_feat_saver = []
        for v_emb, visual_guided_att, visual_reduce in zip(self.v_embs, self.visual_guided_atts, self.visual_reduces):
            v_embed = v_emb(v)
            v_feat_saver.append(v_embed)
            v_guided = visual_guided_att(v_embed, q_feat)
            v_feats.append(visual_reduce(v_guided, v_embed))

        
        v_feat_vit = v_feat_saver[0]
        v_feat_cnn = v_feat_saver[1]


        v_joint_feat = torch.cat(v_feats, dim=1)
        v_joint_feat = v_joint_feat.unsqueeze(1)


        q_feat = self.q_guided_att(q_feat, v_joint_feat)
        # print("q_feat:",q_feat.shape)
        q_feat = q_feat.mean(1)

        out = self.fusion(q_feat, v_joint_feat.squeeze(1))

        out = self.classifier(out)

        ####################################### TOTAL ##########################



        out = out*(gate_output[:,0].view(-1,1))
        zero_classes_onehot = zero_classes_onehot * gate_output[:,1].view(-1,1)
        
        sum_out = out + zero_classes_onehot

        return sum_out
    
    def classify(self, x):
        return self.classifier(x)


def build_GuidedAtt(args):
    print('Use BERT as question embedding...')
    q_dim = args.q_dim
    q_emb = BertQuestionEmbedding(args.bert_pretrained, args.device, use_mhsa=True)
    utils.set_parameters_requires_grad(q_emb, True)

    print('Loading Vision Transformer feature extractor...')
    v_vit_dim = args.v_vit_dim
    v_vit_emb = initialize_backbone_model(args.vit_backbone, use_imagenet_pretrained=args.vit_image_pretrained)[0]

    print(f'Loading CNN ({args.cnn_image_pretrained}) feature extractor...')
    v_cnn_dim = args.v_cnn_dim
    v_cnn_emb = initialize_backbone_model(args.cnn_image_pretrained, use_imagenet_pretrained=True)[0]
    
    # args.v_common_dim = v_common_dim = v_vit_dim
    cnn_converter = nn.Linear(v_cnn_dim, v_vit_dim)
    v_cnn_emb = nn.Sequential(v_cnn_emb, cnn_converter)
    v_cnn_dim = v_vit_dim
    
    visual_vit_guided_att = GuidedTransformerEncoder(v_vit_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)
    visual_cnn_guided_att = GuidedTransformerEncoder(v_cnn_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)

    visual_vit_reduced = AttentionReduce(v_vit_dim, v_vit_dim // 2, args.glimpse)
    visual_cnn_reduced = AttentionReduce(v_cnn_dim, v_cnn_dim // 2, args.glimpse)    

    visual_bbox_guided_att = None
    
    # question_guided_att = GuidedTransformerEncoder(q_dim, 1024, args.num_heads, args.hidden_dim, args.dropout)
    question_guided_att = GuidedTransformerEncoder(q_dim, v_vit_dim + v_cnn_dim, args.num_heads, args.hidden_dim, args.dropout)

    fusion = FusionLinear(q_dim, v_vit_dim + v_cnn_dim, args.joint_dim)

    classifier = Classifier(
        in_features=(args.joint_dim,q_dim),
        mid_features=args.joint_dim * 2,
        out_features=args.num_classes,
        count_features=args.objects + 1,
        drop=args.dropout,
    )

    return GuidedAttentionModel(
        q_emb, 
        [v_vit_emb, v_cnn_emb], 
        [visual_vit_guided_att, visual_cnn_guided_att],
        [visual_vit_reduced, visual_cnn_reduced],
        fusion,
        question_guided_att,
        classifier,
        visual_bbox_guided_att,
        args
    )
    
def build_GuidedAtt_replaceResNetViT(args, replacingModel, replacingModelViT):
    print('Use BERT as question embedding...')
    q_dim = args.q_dim
    q_emb = BertQuestionEmbedding(args.bert_pretrained, args.device, use_mhsa=True)
    utils.set_parameters_requires_grad(q_emb, False) 

    ########## Replace this ############
    print('Loading Vision Transformer feature extractor...')
    v_vit_dim = args.v_vit_dim
    v_vit_emb = replacingModelViT
    ########## Replace this ############

    ########## Replace this ############
    print(f'Loading CNN ({args.cnn_image_pretrained}) feature extractor...')
    v_cnn_dim = args.v_cnn_dim
    v_cnn_emb = replacingModel
     ########## Replace this ############
    # args.v_common_dim = v_common_dim = v_vit_dim
    # print("v_cnn_dim:",v_cnn_dim)
    # print("v_vit_dim:",v_vit_dim)
    cnn_converter = nn.Linear(v_cnn_dim, v_vit_dim)
    v_cnn_emb = nn.Sequential(v_cnn_emb, cnn_converter)
    v_cnn_dim = v_vit_dim
    
    visual_vit_guided_att = GuidedTransformerEncoder(v_vit_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)
    visual_cnn_guided_att = GuidedTransformerEncoder(v_cnn_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)

    visual_vit_reduced = AttentionReduce(v_vit_dim, v_vit_dim // 2, args.glimpse)
    visual_cnn_reduced = AttentionReduce(v_cnn_dim, v_cnn_dim // 2, args.glimpse)    

    # visual_bbox_guided_att = GuidedTransformerEncoder(1, 100, args.num_heads, args.hidden_dim, args.dropout)
    visual_bbox_guided_att = None
    
    # question_guided_att = GuidedTransformerEncoder(q_dim, 1024, args.num_heads, args.hidden_dim, args.dropout)
    question_guided_att = GuidedTransformerEncoder(q_dim, v_vit_dim + v_cnn_dim, args.num_heads, args.hidden_dim, args.dropout)

    fusion = FusionLinear(q_dim, v_vit_dim + v_cnn_dim, args.joint_dim)


    ############################################################# OD feature additional #################
    od_visual_vit_guided_att = GuidedTransformerEncoder(100, q_dim, 4, args.hidden_dim, args.dropout)
    od_question_guided_att = GuidedTransformerEncoder(q_dim, 100, 4, args.hidden_dim, args.dropout)
    od_fusion = FusionLinear(q_dim, 100, 100)
    #####################################################################################################

    classifier = SimpleClassifier(
        args.joint_dim, args.joint_dim * 2, args.num_classes, args)

    # classifier = Classifier(
    #     in_features=(args.joint_dim,q_dim),
    #     mid_features=args.joint_dim * 2,
    #     out_features=args.num_classes,
    #     count_features=args.objects + 1,
    #     drop=args.dropout,
    # )

    return GuidedAttentionModel(
        q_emb, 
        [v_vit_emb, v_cnn_emb], 
        [visual_vit_guided_att, visual_cnn_guided_att],
        [visual_vit_reduced, visual_cnn_reduced],
        fusion,
        question_guided_att,
        classifier,
        od_visual_vit_guided_att,
        od_question_guided_att,
        od_fusion,
        args
    )



class CountingModule(nn.Module):
    def __init__(self, q_emb,  od_visual_vit_guided_att, od_question_guided_att, OD_model, OD_feature_extractor, OD_image_processor, classifier, OD_model_get, args):
        super(CountingModule, self).__init__()
        self.objects = 9
        self.q_emb = q_emb
        
        self.OD_model, self.OD_feature_extractor, self.OD_image_processor = OD_model, OD_feature_extractor, OD_image_processor

        self.od_visual_vit_guided_att = od_visual_vit_guided_att
        self.od_question_guided_att = od_question_guided_att
        
        self.num_classes =  args.num_classes
        self.gpu = args.gpu
        self.counting_mode = args.counting_mode
        self.is_using_guilded = args.is_using_guilded
        self.fusion_mode = args.fusion_mode
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and (args.gpu != "-1") else "cpu")
        if self.counting_mode == 1:
            self.counter = Counter(self.objects)
        if self.counting_mode == 1 or self.counting_mode == 2: 
            self.od_fusion = FusionLinear(args.q_dim, 100, 100)
        self.ID_number = {
            1: 323,
            2: 329,
            3: 26,
            4: 46,
            5: 106,
            6: 214,
            7: 312,
            8: 209,
            9: 201,
            10: 263
        }
        # 1 object -> "một" -> id in dataset = 323

        # self.zero_classes_onehot = torch.zeros((args.batch_size, self.num_classes))
        # self.zero_classes_onehot = self.zero_classes_onehot.to("cuda:0")
        self.classifier = classifier
        self.OD_model_get = OD_model_get
        if args.OD_model_get == "logits":
            input_v_dim_fusion = 100*92
        else:
            input_v_dim_fusion = 100*4
        input_q_dim_fusion = args.q_dim*args.question_len
        fusion_dim = 256
        input_of_classifier_dim =128

        if self.counting_mode == 3:
            # change fusion module
            if self.is_using_guilded == 1:
                if self.fusion_mode == 'add':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim, input_of_classifier_dim)
                    

                elif self.fusion_mode == 'concat':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim + fusion_dim, input_of_classifier_dim)

                elif self.fusion_mode == 'mul':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim, input_of_classifier_dim)

            else:
                if self.fusion_mode == 'add':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim, input_of_classifier_dim)
                   
                elif self.fusion_mode == 'concat':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim + fusion_dim, input_of_classifier_dim)
                elif self.fusion_mode == 'mul':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim, input_of_classifier_dim)
        elif self.counting_mode == 4:
            self.SiLU = torch.nn.SiLU()
            input_of_classifier_dim = 100
            if self.is_using_guilded == 1:
                if self.fusion_mode == 'add':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim, input_of_classifier_dim)
                    self.dense2onehot = SimpleClassifier(input_of_classifier_dim, input_of_classifier_dim * 2, args.num_classes, args)

                elif self.fusion_mode == 'concat':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim + fusion_dim, input_of_classifier_dim)
                    self.dense2onehot = SimpleClassifier(input_of_classifier_dim, input_of_classifier_dim * 2, args.num_classes, args)
                elif self.fusion_mode == 'mul':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim, input_of_classifier_dim)
                    self.dense2onehot = SimpleClassifier(input_of_classifier_dim, input_of_classifier_dim * 2, args.num_classes, args)

            else:
                if self.fusion_mode == 'add':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim, input_of_classifier_dim)
                    self.dense2onehot = SimpleClassifier(input_of_classifier_dim, input_of_classifier_dim * 2, args.num_classes, args)
                   
                elif self.fusion_mode == 'concat':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim + fusion_dim, input_of_classifier_dim)
                    self.dense2onehot = SimpleClassifier(input_of_classifier_dim, input_of_classifier_dim * 2, args.num_classes, args)
                elif self.fusion_mode == 'mul':
                    self.pre_fusion_v = nn.Linear(input_v_dim_fusion, fusion_dim)
                    self.pre_fusion_q = nn.Linear(input_q_dim_fusion, fusion_dim)
                    self.dense = nn.Linear(fusion_dim, input_of_classifier_dim)
                    self.dense2onehot = SimpleClassifier(input_of_classifier_dim, input_of_classifier_dim * 2, args.num_classes, args)

            
    def forward(self, q, img, raw_img_list):

        ############ get bboxes ####################################
        OD_inputs = self.OD_feature_extractor(images=raw_img_list, return_tensors="pt")
        OD_inputs = OD_inputs.to(self.device)
        OD_outputs = self.OD_model(**OD_inputs)
        list_box = torch.permute(OD_outputs[self.OD_model_get].detach() , (0, 2, 1))
        b = list_box.to(self.device)

        ############################################################

        q_feat = self.q_emb(q)

        ############################################################
        if self.counting_mode == 1:
            od_v_guided = self.od_visual_vit_guided_att(b, q_feat)   
            od_q_guided = self.od_question_guided_att(q_feat, b)     
            a = self.od_fusion(od_q_guided.mean(1), od_v_guided.mean(1))
            one_hot_conf = self.counter(b, a)
            return one_hot_conf
        elif self.counting_mode == 2:
            od_v_guided = self.od_visual_vit_guided_att(b, q_feat)   
            od_q_guided = self.od_question_guided_att(q_feat, b)     
            a = self.od_fusion(od_q_guided.mean(1), od_v_guided.mean(1))
            one_hot_conf = self.classifier(a)
            return one_hot_conf
        
        elif self.counting_mode == 3:
            # change fusion module
            if self.is_using_guilded == 1:
                od_v_guided = self.od_visual_vit_guided_att(b, q_feat)   
                od_q_guided = self.od_question_guided_att(q_feat, b)     
                # a = self.od_fusion(od_q_guided.mean(1), od_v_guided.mean(1))
                if self.fusion_mode == 'add':
                    od_v_guided = od_v_guided.reshape(q_feat.shape[0], -1) #batch x (len_v * dim_v)
                    od_q_guided = od_q_guided.reshape(q_feat.shape[0], -1) #batch x (len_q * dim_q)
                    od_v_guided = self.pre_fusion_v(od_v_guided)
                    od_q_guided = self.pre_fusion_q(od_q_guided)
                    a = torch.add(od_v_guided, od_q_guided)
                    a = torch.nn.functional.relu(self.dense(a))
                    od_v_guided = None
                    od_q_guided = None

                    

                elif self.fusion_mode == 'concat':
                    od_v_guided = od_v_guided.reshape(q_feat.shape[0], -1) #batch x (len_v * dim_v)
                    od_q_guided = od_q_guided.reshape(q_feat.shape[0], -1) #batch x (len_q * dim_q)
                    od_v_guided = self.pre_fusion_v(od_v_guided)
                    od_q_guided = self.pre_fusion_q(od_q_guided)
                    a = torch.concat((od_v_guided, od_q_guided), -1)
                    a = self.dense(a)
                    a = torch.nn.functional.relu(a)
                elif self.fusion_mode == 'mul':
                    od_v_guided = od_v_guided.reshape(q_feat.shape[0], -1) #batch x (len_v * dim_v)
                    od_q_guided = od_q_guided.reshape(q_feat.shape[0], -1) #batch x (len_q * dim_q)
                    od_v_guided = self.pre_fusion_v(od_v_guided)
                    od_q_guided = self.pre_fusion_q(od_q_guided)
                    a = torch.multiply(od_v_guided, od_q_guided)
                    a = self.dense(a)
                    a = torch.nn.functional.relu(a)

            else:
                if self.fusion_mode == 'add':
                    b = b.reshape(q_feat.shape[0], -1); od_v = self.pre_fusion_v(b)
                    q_feat = q_feat.reshape(q_feat.shape[0], -1); od_q = self.pre_fusion_q(q_feat)
                    a = torch.add(od_v, od_q)
                    a = self.dense(a)
                    a = torch.nn.functional.relu(a)
                elif self.fusion_mode == 'concat':
                    b = b.reshape(q_feat.shape[0], -1); od_v = self.pre_fusion_v(b)
                    q_feat = q_feat.reshape(q_feat.shape[0], -1); od_q = self.pre_fusion_q(q_feat)
                    a = torch.concat((od_v, od_q),-1)
                    a = self.dense(a)
                    a = torch.nn.functional.relu(a)
                elif self.fusion_mode == 'mul':
                    b = b.reshape(q_feat.shape[0], -1); od_v = self.pre_fusion_v(b)
                    q_feat = q_feat.reshape(q_feat.shape[0], -1); od_q = self.pre_fusion_q(q_feat)
                    a = torch.multiply(od_v, od_q)
                    a = self.dense(a)
                    a = torch.nn.functional.relu(a)



            one_hot_conf = self.classifier(a)
            # a = None
            # b = None
            # list_box = None
            # q_feat = None
            # q = None
            # OD_inputs = None
            # OD_outputs = None
            del OD_inputs
            del OD_outputs
            del list_box
            del b
            return one_hot_conf

        elif self.counting_mode == 4:
            #count bbox conf
            if self.is_using_guilded == 1:
                od_v_guided = self.od_visual_vit_guided_att(b, q_feat)   
                od_q_guided = self.od_question_guided_att(q_feat, b)   
                if self.fusion_mode == 'add':
                    od_v_guided = od_v_guided.reshape(q_feat.shape[0], -1) #batch x (len_v * dim_v)
                    od_q_guided = od_q_guided.reshape(q_feat.shape[0], -1) #batch x (len_q * dim_q)
                    od_v_guided = self.SiLU(self.pre_fusion_v(od_v_guided))
                    od_q_guided = self.SiLU(self.pre_fusion_q(od_q_guided))
                    a = torch.add(od_v_guided, od_q_guided)
                    a = self.dense(a)
                    one_hot = self.dense2onehot(a)
                    a = torch.sigmoid(a)   #batch x bbox_number
                    a = torch.sum(a, dim=-1) #batch x 1   range [0,100]-> scale [1,10]
                    a = (a/100) * 9 + 1
                    del OD_inputs
                    del OD_outputs
                    del list_box
                    del b
                    return torch.round(a).clamp(1,10), one_hot
                elif self.fusion_mode == 'concat':
                    od_v_guided = od_v_guided.reshape(q_feat.shape[0], -1) #batch x (len_v * dim_v)
                    od_q_guided = od_q_guided.reshape(q_feat.shape[0], -1) #batch x (len_q * dim_q)
                    od_v_guided = self.SiLU(self.pre_fusion_v(od_v_guided))
                    od_q_guided = self.SiLU(self.pre_fusion_q(od_q_guided))
                    a = torch.concat((od_v_guided, od_q_guided), -1)
                    a = self.dense(a)
                    one_hot = self.dense2onehot(a)
                    a = torch.sigmoid(a)   #batch x bbox_number
                    a = torch.sum(a, dim=-1) #batch x 1
                    a = (a/100) * 9 + 1
                    del OD_inputs
                    del OD_outputs
                    del list_box
                    del b
                    return torch.round(a).clamp(1,10), one_hot
                elif self.fusion_mode == 'mul':
                    od_v_guided = od_v_guided.reshape(q_feat.shape[0], -1) #batch x (len_v * dim_v)
                    od_q_guided = od_q_guided.reshape(q_feat.shape[0], -1) #batch x (len_q * dim_q)
                    od_v_guided = self.SiLU(self.pre_fusion_v(od_v_guided))
                    od_q_guided = self.SiLU(self.pre_fusion_q(od_q_guided))
                    a = torch.multiply(od_v_guided, od_q_guided)
                    a = self.dense(a)
                    one_hot = self.dense2onehot(a)
                    a = torch.sigmoid(a)   #batch x bbox_number
                    a = torch.sum(a, dim=-1) #batch x 1
                    a = (a/100) * 9 + 1
                    del OD_inputs
                    del OD_outputs
                    del list_box
                    del b
                    return torch.round(a).clamp(1,10), one_hot
            else:
                if self.fusion_mode == 'add':
                    b = b.reshape(q_feat.shape[0], -1)
                    od_v = self.SiLU(self.pre_fusion_v(b))
                    q_feat = q_feat.reshape(q_feat.shape[0], -1)
                    od_q = self.SiLU(self.pre_fusion_q(q_feat))
                    a = torch.add(od_v, od_q)
                    a = self.dense(a)
                    one_hot = self.dense2onehot(a)
                    a = torch.sigmoid(a)   #batch x bbox_number
                    a = torch.sum(a, dim=-1) #batch x 1
                    a = (a/100) * 9 + 1
                    del OD_inputs
                    del OD_outputs
                    del list_box
                    del b
                    return torch.round(a).clamp(1,10), one_hot
                elif self.fusion_mode == 'concat':
                    b = b.reshape(q_feat.shape[0], -1)
                    od_v = self.SiLU(self.pre_fusion_v(b))
                    q_feat = q_feat.reshape(q_feat.shape[0], -1)
                    od_q = self.SiLU(self.pre_fusion_q(q_feat))
                    a = torch.concat((od_v, od_q),-1)
                    a = self.dense(a)
                    one_hot = self.dense2onehot(a)
                    a = torch.sigmoid(a)   #batch x bbox_number
                    a = torch.sum(a, dim=-1) #batch x 1
                    a = (a/100) * 9 + 1
                    del OD_inputs
                    del OD_outputs
                    del list_box
                    del b
                    return torch.round(a).clamp(1,10), one_hot
                elif self.fusion_mode == 'mul':
                    b = b.reshape(q_feat.shape[0], -1)
                    od_v = self.SiLU(self.pre_fusion_v(b))
                    q_feat = q_feat.reshape(q_feat.shape[0], -1)
                    od_q = self.SiLU(self.pre_fusion_q(q_feat))
                    a = torch.multiply(od_v, od_q)
                    a = self.dense(a)
                    one_hot = self.dense2onehot(a)
                    a = torch.sigmoid(a)   #batch x bbox_number
                    a = torch.sum(a, dim=-1) #batch x 1
                    a = (a/100) * 9 + 1
                    del OD_inputs
                    del OD_outputs
                    del list_box
                    del b
                    return torch.round(a).clamp(1,10), one_hot



    
def build_CountingModule(args):
    print('Use BERT as question embedding...')
    q_dim = args.q_dim
    q_emb = BertQuestionEmbedding(args.bert_pretrained, args.device, use_mhsa=True)
    
    if not args.freeze_bert:
        q_emb = utils.set_parameters_requires_grad(q_emb, True)
    else:
        q_emb = utils.set_parameters_requires_grad(q_emb, False)

    ############################################################# OD feature additional #################
    od_visual_vit_guided_att = GuidedTransformerEncoder(100, q_dim, 4, args.hidden_dim, args.dropout)
    od_question_guided_att = GuidedTransformerEncoder(q_dim, 100, 4, args.hidden_dim, args.dropout)
    
    #####################################################################################################
    


    ############################################################# OD get bboxes #########################
    OD_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    OD_feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-tiny')
    OD_model = utils.set_parameters_requires_grad(OD_model, False)
    OD_image_processor = None
    #####################################################################################################

    classifier = SimpleClassifier(
        128, 128 * 2, args.num_classes, args)

    OD_model_get = args.OD_model_get

    return CountingModule(
        q_emb, 
        od_visual_vit_guided_att,
        od_question_guided_att,
        OD_model, OD_feature_extractor, OD_image_processor,
        classifier,OD_model_get,
        args
    )