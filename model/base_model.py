"""
This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""

import torch
import torch.nn as nn
# from attention import BiAttention, StackedAttention
from model.co_attention import CoTransformerBlock, FusionAttentionFeature, GuidedTransformerEncoder, AttentionReduce, FusionLinear
from language_model import WordEmbedding, QuestionEmbedding, BertQuestionEmbedding, SelfAttention
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
# from counting import Counter
# from utils import tfidf_loading, generate_spatial_batch
# from simple_cnn import SimpleCNN
# from auto_encoder import Auto_Encoder_Model
from backbone import initialize_backbone_model, ObjectDetectionModel
# from multi_task import ResNet50, ResNet18, ResNet34
# from mc import MCNet
# from convert import Convert, GAPConvert
import os
# from non_local import NONLocalBlock3D
# from transformer.SubLayers import MultiHeadAttention
import utils_VQA as utils




class GuidedAttentionModel(nn.Module):
    def __init__(self, q_emb, v_embs, visual_guided_atts, visual_reduces, fusion, q_guided_att, classifier, args):
        super(GuidedAttentionModel, self).__init__()
        self.q_emb = q_emb
        
        self.v_embs = nn.ModuleList(v_embs)
        self.visual_guided_atts = nn.ModuleList(visual_guided_atts)
        self.visual_reduces = nn.ModuleList(visual_reduces)
        
        self.fusion = fusion
        self.q_guided_att = q_guided_att
        self.question_reduced = AttentionReduce(768, 768 // 2, 1)

        self.classifier = classifier
        self.flatten = nn.Flatten()
    
    def forward(self, v, q):
        q_feat = self.q_emb(q)
        v_embed = None
        v_feat_cnn = None
        v_feat_vit = None
        v_feats = []
        for v_emb, visual_guided_att, visual_reduce in zip(self.v_embs, self.visual_guided_atts, self.visual_reduces):
            v_embed = v_emb(v)
            
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print("v_embed:",v_embed)
            # print("v_embed.shape:",v_embed.shape)
            # exit()
            v_guided = visual_guided_att(v_embed, q_feat)
            
            v_feats.append(visual_reduce(v_guided, v_embed))


            # print("v_guided:",v_guided.shape)
            # v_feats.append(visual_reduce(v_embed, v_embed))
            # v_feats.append(v_guided.mean(1, keepdim=True))
        v_feat_vit = v_feats[0]
        v_feat_cnn = v_feats[1]
        # print("v_feats[0]:",v_feats[0].shape) #batch x 768
        # print("v_feats[1]:",v_feats[1].shape) #batch x 768
        # exit()
        # v_joint_feat = self.fusion(*v_feats)
        
        # v_joint_feat = torch.mul(*v_feats)
        # v_joint_feat = torch.stack(v_feats, dim=-1).sum(-1)
        v_joint_feat = torch.cat(v_feats, dim=1)
        v_joint_feat = v_joint_feat.unsqueeze(1)
        
        # out = out.mean(1, keepdim =True) # average pooling
        # out = self.flatten(out)

        # v_joint_feat = torch.cat(v_feats, dim=1)
        # v_joint_feat = v_joint_feat.unsqueeze(1)

        q_feat = self.q_guided_att(q_feat, v_joint_feat)
        q_feat = q_feat.mean(1)
        # out = self.question_reduced(q_feat, q_feat)
        
        out = self.fusion(q_feat, v_joint_feat.squeeze(1))
        #out = self.classifier(out)
        return out, v_feat_cnn, v_feat_vit
    
    def classify(self, x):
        return self.classifier(x)
# class GuidedAttentionModel(nn.Module):
#     def __init__(self, q_emb, v_embs, visual_guided_atts, visual_reduces, fusion, q_guided_att, classifier, args):
#         super(GuidedAttentionModel, self).__init__()
#         self.q_emb = q_emb
        
#         self.v_embs = nn.ModuleList(v_embs)
#         self.visual_guided_atts = nn.ModuleList(visual_guided_atts)
#         self.visual_reduces = nn.ModuleList(visual_reduces)
        
#         self.fusion = fusion
#         self.q_guided_att = q_guided_att
#         self.question_reduced = AttentionReduce(768, 768 // 2, 1)

#         self.classifier = classifier
#         self.flatten = nn.Flatten()
    
#     def forward(self, v, q):
#         q_feat = self.q_emb(q)

#         v_feats = []
#         for v_emb, visual_guided_att, visual_reduce in zip(self.v_embs, self.visual_guided_atts, self.visual_reduces):
#             v_embed = v_emb(v)
#             v_guided = visual_guided_att(v_embed, q_feat)
            
#             v_feats.append(visual_reduce(v_guided, v_embed))
#             # v_feats.append(visual_reduce(v_embed, v_embed))
#             # v_feats.append(v_guided.mean(1, keepdim=True))

#         # v_joint_feat = self.fusion(*v_feats)
        
#         # v_joint_feat = torch.mul(*v_feats)
#         # v_joint_feat = torch.stack(v_feats, dim=-1).sum(-1)
#         v_joint_feat = torch.cat(v_feats, dim=1)
#         v_joint_feat = v_joint_feat.unsqueeze(1)
        
#         # out = out.mean(1, keepdim =True) # average pooling
#         # out = self.flatten(out)

#         # v_joint_feat = torch.cat(v_feats, dim=1)
#         # v_joint_feat = v_joint_feat.unsqueeze(1)

#         q_feat = self.q_guided_att(q_feat, v_joint_feat)
#         q_feat = q_feat.mean(1)
#         # out = self.question_reduced(q_feat, q_feat)
        
#         out = self.fusion(q_feat, v_joint_feat.squeeze(1))
#         # out = self.classifier(out)
#         return out
    
#     def classify(self, x):
#         return self.classifier(x)

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
    
    # question_guided_att = GuidedTransformerEncoder(q_dim, 1024, args.num_heads, args.hidden_dim, args.dropout)
    question_guided_att = GuidedTransformerEncoder(q_dim, v_vit_dim + v_cnn_dim, args.num_heads, args.hidden_dim, args.dropout)

    fusion = FusionLinear(q_dim, v_vit_dim + v_cnn_dim, args.joint_dim)

    classifier = SimpleClassifier(
        args.joint_dim, args.joint_dim * 2, args.num_classes, args)

    return GuidedAttentionModel(
        q_emb, 
        [v_vit_emb, v_cnn_emb], 
        [visual_vit_guided_att, visual_cnn_guided_att],
        [visual_vit_reduced, visual_cnn_reduced],
        fusion,
        question_guided_att,
        classifier,
        args
    )
    
def build_GuidedAtt_replaceResNet(args, replacingModel):
    print('Use BERT as question embedding...')
    q_dim = args.q_dim
    q_emb = BertQuestionEmbedding(args.bert_pretrained, args.device, use_mhsa=True)
    utils.set_parameters_requires_grad(q_emb, True)

    print('Loading Vision Transformer feature extractor...')
    v_vit_dim = args.v_vit_dim
    v_vit_emb = initialize_backbone_model(args.vit_backbone, use_imagenet_pretrained=args.vit_image_pretrained)[0]

    ########## Replace this ############
    print(f'Loading CNN ({args.cnn_image_pretrained}) feature extractor...')
    v_cnn_dim = args.v_cnn_dim
    v_cnn_emb = replacingModel
     ########## Replace this ############
    # args.v_common_dim = v_common_dim = v_vit_dim
    # print("v_cnn_dim:",v_cnn_dim)
    # print("v_vit_dim:",v_vit_dim)
    # exit()
    cnn_converter = nn.Linear(v_cnn_dim, v_vit_dim)
    v_cnn_emb = nn.Sequential(v_cnn_emb, cnn_converter)
    v_cnn_dim = v_vit_dim
    
    visual_vit_guided_att = GuidedTransformerEncoder(v_vit_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)
    visual_cnn_guided_att = GuidedTransformerEncoder(v_cnn_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)

    visual_vit_reduced = AttentionReduce(v_vit_dim, v_vit_dim // 2, args.glimpse)
    visual_cnn_reduced = AttentionReduce(v_cnn_dim, v_cnn_dim // 2, args.glimpse)    
    
    # question_guided_att = GuidedTransformerEncoder(q_dim, 1024, args.num_heads, args.hidden_dim, args.dropout)
    question_guided_att = GuidedTransformerEncoder(q_dim, v_vit_dim + v_cnn_dim, args.num_heads, args.hidden_dim, args.dropout)

    fusion = FusionLinear(q_dim, v_vit_dim + v_cnn_dim, args.joint_dim)

    classifier = SimpleClassifier(
        args.joint_dim, args.joint_dim * 2, args.num_classes, args)

    return GuidedAttentionModel(
        q_emb, 
        [v_vit_emb, v_cnn_emb], 
        [visual_vit_guided_att, visual_cnn_guided_att],
        [visual_vit_reduced, visual_cnn_reduced],
        fusion,
        question_guided_att,
        classifier,
        args
    )

def build_GuidedAtt_replaceResNetViT(args, replacingModel, replacingModelViT):
    print('Use BERT as question embedding...')
    q_dim = args.q_dim
    q_emb = BertQuestionEmbedding(args.bert_pretrained, args.device, use_mhsa=True)
    utils.set_parameters_requires_grad(q_emb, True)

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
    # exit()
    cnn_converter = nn.Linear(v_cnn_dim, v_vit_dim)
    v_cnn_emb = nn.Sequential(v_cnn_emb, cnn_converter)
    v_cnn_dim = v_vit_dim
    
    visual_vit_guided_att = GuidedTransformerEncoder(v_vit_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)
    visual_cnn_guided_att = GuidedTransformerEncoder(v_cnn_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)

    visual_vit_reduced = AttentionReduce(v_vit_dim, v_vit_dim // 2, args.glimpse)
    visual_cnn_reduced = AttentionReduce(v_cnn_dim, v_cnn_dim // 2, args.glimpse)    
    
    # question_guided_att = GuidedTransformerEncoder(q_dim, 1024, args.num_heads, args.hidden_dim, args.dropout)
    question_guided_att = GuidedTransformerEncoder(q_dim, v_vit_dim + v_cnn_dim, args.num_heads, args.hidden_dim, args.dropout)

    fusion = FusionLinear(q_dim, v_vit_dim + v_cnn_dim, args.joint_dim)

    classifier = SimpleClassifier(
        args.joint_dim, args.joint_dim * 2, args.num_classes, args)

    return GuidedAttentionModel(
        q_emb, 
        [v_vit_emb, v_cnn_emb], 
        [visual_vit_guided_att, visual_cnn_guided_att],
        [visual_vit_reduced, visual_cnn_reduced],
        fusion,
        question_guided_att,
        classifier,
        args
    )
