"""
This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import torch
import torch.nn as nn
import torchvision
from attention import BiAttention, StackedAttention
from co_attention import CoTransformerBlock, FusionAttentionFeature, GuidedTransformerEncoder, AttentionReduce, FusionLinear
from language_model import WordEmbedding, QuestionEmbedding, BertQuestionEmbedding, SelfAttention
from classifier import SimpleClassifier, Classifier
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
    
class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus ReLU'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.ReLU(x + y)

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


class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
        self.q_lin = nn.Linear(q_features, mid_features)
        self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
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
    def __init__(self, q_emb, v_embs, visual_guided_atts, visual_reduces, fusion, q_guided_att, classifier, args):
        super(GuidedAttentionModel, self).__init__()
        # self.objects = args.objects
        self.q_emb = q_emb
        
        self.v_embs = nn.ModuleList(v_embs)
        self.visual_guided_atts = nn.ModuleList(visual_guided_atts)
        self.visual_reduces = nn.ModuleList(visual_reduces)
        
        self.fusion = fusion
        self.q_guided_att = q_guided_att
        # self.question_reduced = AttentionReduce(768, 768 // 2, 1)

        self.classifier = classifier
        # self.flatten = nn.Flatten()
        # self.counter = Counter(self.objects)
        # self.last_linear = nn.Linear(args.joint_dim, args.joint_dim//2)
        # self.attention = Attention(
        #     v_features=args.v_vit_dim,
        #     q_features=args.q_dim,
        #     mid_features=512,
        #     glimpses=args.glimpse,
        #     drop=0.5,
        # )
        # self.SiLU = torch.nn.SiLU()
    
    def forward(self, v, q):
        q_feat = self.q_emb(q)

        v_feats = []
        v_feat_saver = []
        for v_emb, visual_guided_att, visual_reduce in zip(self.v_embs, self.visual_guided_atts, self.visual_reduces):
            v_embed = v_emb(v)
            v_feat_saver.append(v_embed)
            v_guided = visual_guided_att(v_embed, q_feat)
            # print("visual_reduce(v_guided, v_embed).shape:",visual_reduce(v_guided, v_embed).shape)
            v_feats.append(visual_reduce(v_guided, v_embed))
            # v_feats.append(visual_reduce(v_embed, v_embed))
            # v_feats.append(v_guided.mean(1, keepdim=True))
        
        v_feat_vit = v_feat_saver[0]
        v_feat_cnn = v_feat_saver[1]
        # this is where the counting component is used
        # pick out the first attention map
        # a = self.attention(v_feats[0],q_feat)

        # a1 = a[:, 0, :, :].contiguous().view(a.size(0), -1)
        # give it and the bounding boxes to the component
        # self.counter(b, a1)
        # v_joint_feat = self.fusion(*v_feats)
        
        # v_joint_feat = torch.mul(*v_feats)
        # v_joint_feat = torch.stack(v_feats, dim=-1).sum(-1)
        v_joint_feat = torch.cat(v_feats, dim=1)
        v_joint_feat = v_joint_feat.unsqueeze(1)
        # print(v_joint_feat.shape)
        # exit()
        
        # out = out.mean(1, keepdim =True) # average pooling
        # out = self.flatten(out)

        # v_joint_feat = torch.cat(v_feats, dim=1)
        # v_joint_feat = v_joint_feat.unsqueeze(1)

        q_feat = self.q_guided_att(q_feat, v_joint_feat)
        q_feat = q_feat.mean(1)
        # out = self.question_reduced(q_feat, q_feat)
        
        out = self.fusion(q_feat, v_joint_feat.squeeze(1))
        # out = self.SiLU(self.last_linear(out))
        #out = self.classifier(out)
        return out, v_feat_vit, v_feat_cnn
    
    def classify(self, x):
        return self.classifier(x)

class DucAnhGuidedAttentionModel(nn.Module):
    def __init__(self, q_emb, v_embs, visual_guided_atts, visual_reduces, fusion, q_guided_att, classifier, args):
        super(DucAnhGuidedAttentionModel, self).__init__()
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

        v_feats = []
        for v_emb, visual_guided_att, visual_reduce in zip(self.v_embs, self.visual_guided_atts, self.visual_reduces):
            v_embed = v_emb(v)
            v_guided = visual_guided_att(v_embed, q_feat)
            
            v_feats.append(visual_reduce(v_guided, v_embed))
            # v_feats.append(visual_reduce(v_embed, v_embed))
            # v_feats.append(v_guided.mean(1, keepdim=True))

        v_feat_vit = v_feats[0]
        v_feat_cnn = v_feats[1]

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
        return out, v_feat_vit, v_feat_cnn
    
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
    
    # question_guided_att = GuidedTransformerEncoder(q_dim, 1024, args.num_heads, args.hidden_dim, args.dropout)
    question_guided_att = GuidedTransformerEncoder(q_dim, v_vit_dim + v_cnn_dim, args.num_heads, args.hidden_dim, args.dropout)

    fusion = FusionLinear(q_dim, v_vit_dim + v_cnn_dim, args.joint_dim)

    classifier = SimpleClassifier(
        args.joint_dim, args.joint_dim*2 , args.num_classes, args)

    return DucAnhGuidedAttentionModel(
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
        args.joint_dim, args.joint_dim*2 , args.num_classes, args)

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
        args
    )






class GuidedAttentionModelFastRCNN(nn.Module):
    def __init__(self, q_emb, v_embs, visual_guided_atts, visual_reduces, fusion, q_guided_att, classifier, fasterrcnn_mobilenet, args):
        super(GuidedAttentionModelFastRCNN, self).__init__()
        self.objects = args.objects
        self.q_emb = q_emb
        
        self.v_embs = nn.ModuleList(v_embs)
        self.visual_guided_atts = nn.ModuleList(visual_guided_atts)
        self.visual_reduces = nn.ModuleList(visual_reduces)
        
        self.fusion = fusion
        self.reduce_last_hid_detection_layer_conv = nn.Conv2d(960, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=2)
        self.fusion_v2 = FusionLinear(args.joint_dim, 9*9*128, args.joint_dim)
        self.q_guided_att = q_guided_att
        self.question_reduced = AttentionReduce(768, 768 // 2, 1)

        self.classifier = classifier
        self.flatten = nn.Flatten()
        self.counter = Counter(self.objects)
        self.last_linear = nn.Linear(args.joint_dim, args.joint_dim//2)
        self.attention = Attention(
            v_features=args.v_vit_dim,
            q_features=args.q_dim,
            mid_features=512,
            glimpses=args.glimpse,
            drop=0.5,
        )
        self.SiLU = torch.nn.SiLU()
        self.fasterrcnn_mobilenet = fasterrcnn_mobilenet
        
    
    def forward(self, v, q):
        
        self.fasterrcnn_mobilenet.eval()
        q_feat = self.q_emb(q)

        v_feats = []
        v_feat_saver = []
        for v_emb, visual_guided_att, visual_reduce in zip(self.v_embs, self.visual_guided_atts, self.visual_reduces):
            v_embed = v_emb(v)
            v_feat_saver.append(v_embed)
            v_guided = visual_guided_att(v_embed, q_feat)
            # print("visual_reduce(v_guided, v_embed).shape:",visual_reduce(v_guided, v_embed).shape)
            v_feats.append(visual_reduce(v_guided, v_embed))
            # v_feats.append(visual_reduce(v_embed, v_embed))
            # v_feats.append(v_guided.mean(1, keepdim=True))
        
        v_feat_vit = v_feat_saver[0]
        v_feat_cnn = v_feat_saver[1]

        v_joint_feat = torch.cat(v_feats, dim=1)
        v_joint_feat = v_joint_feat.unsqueeze(1)

        q_feat = self.q_guided_att(q_feat, v_joint_feat)
        q_feat = q_feat.mean(1)
        # out = self.question_reduced(q_feat, q_feat)
        
        out = self.fusion(q_feat, v_joint_feat.squeeze(1))
        fasterrcnn_mobilenet_last_features = []
        def hook(module, input, output):
            fasterrcnn_mobilenet_last_features.append(output['1'].detach())

        handle_hook = self.fasterrcnn_mobilenet.backbone.body.register_forward_hook(hook)
        with torch.no_grad():
            fasterrcnn_mobilenet_out = self.fasterrcnn_mobilenet(v['pixel_values'])
        for temp_out in fasterrcnn_mobilenet_out:
            for k, v in temp_out.items():
                detached_v = v.detach()                  
                temp_out[k] = detached_v

        # fasterrcnn_mobilenet_out["bbox"].detach()
        fasterrcnn_mobilenet_last_hidden_layers = torch.stack(fasterrcnn_mobilenet_last_features).squeeze(0).detach()
        # print("self.fasterrcnn_mobilenet_last_hidden_layers:",self.fasterrcnn_mobilenet_last_hidden_layers.shape)
        fasterrcnn_mobilenet_last_hidden_layers = self.pool(fasterrcnn_mobilenet_last_hidden_layers).detach()
        # print("self.fasterrcnn_mobilenet_last_hidden_layers:",self.fasterrcnn_mobilenet_last_hidden_layers.shape)
        fasterrcnn_mobilenet_last_hidden_layers = self.reduce_last_hid_detection_layer_conv(fasterrcnn_mobilenet_last_hidden_layers).detach()
        # print("fasterrcnn_mobilenet_last_hidden_layers:",fasterrcnn_mobilenet_last_hidden_layers.shape)
        fasterrcnn_mobilenet_last_hidden_layers = fasterrcnn_mobilenet_last_hidden_layers.reshape(fasterrcnn_mobilenet_last_hidden_layers.shape[0],-1)

        out = self.fusion_v2(out, fasterrcnn_mobilenet_last_hidden_layers)


        out = self.SiLU(self.last_linear(out))
        #out = self.classifier(out)
        fasterrcnn_mobilenet_last_hidden_layers = None
        fasterrcnn_mobilenet_out = None
        fasterrcnn_mobilenet_last_features = []
        del fasterrcnn_mobilenet_last_hidden_layers
        del fasterrcnn_mobilenet_out
        handle_hook.remove()
        return out, v_feat_vit, v_feat_cnn
    
    def classify(self, x):
        return self.classifier(x)

def build_GuidedAtt_replaceResNetViT_FastRCNN(args, replacingModel, replacingModelViT):
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
        args.joint_dim//2, args.joint_dim , args.num_classes, args)

    fasterrcnn_mobilenet = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, trainable_backbone_layers=0)
    for param in fasterrcnn_mobilenet.parameters():
        param.requires_grad = False


    return GuidedAttentionModelFastRCNN(
        q_emb, 
        [v_vit_emb, v_cnn_emb], 
        [visual_vit_guided_att, visual_cnn_guided_att],
        [visual_vit_reduced, visual_cnn_reduced],
        fusion,
        question_guided_att,
        classifier, fasterrcnn_mobilenet,
        args
    )

# """
# This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
# """

# import torch
# import torch.nn as nn
# from attention import BiAttention, StackedAttention
# from co_attention import CoTransformerBlock, FusionAttentionFeature, GuidedTransformerEncoder, AttentionReduce, FusionLinear
# from language_model import WordEmbedding, QuestionEmbedding, BertQuestionEmbedding, SelfAttention
# from classifier import SimpleClassifier, Classifier
# from fc import FCNet
# from bc import BCNet
# from counting import Counter
# # from utils import tfidf_loading, generate_spatial_batch
# from simple_cnn import SimpleCNN
# from auto_encoder import Auto_Encoder_Model
# from backbone import initialize_backbone_model, ObjectDetectionModel
# # from multi_task import ResNet50, ResNet18, ResNet34
# from mc import MCNet
# from convert import Convert, GAPConvert
# import os
# from non_local import NONLocalBlock3D
# from transformer.SubLayers import MultiHeadAttention
# import utils



# class CrossAttentionModel(nn.Module):

#     def __init__(self, q_emb, v_emb, co_att_layers, fusion, classifier, args) -> None:
#         super(CrossAttentionModel, self).__init__()
#         self.q_emb = q_emb
#         self.v_emb = v_emb
#         self.classifier = classifier
#         self.co_att_layers = co_att_layers
#         self.fusion = fusion
#         self.flatten = nn.Flatten()
#         self.args = args
        
#     def forward(self, v, q):
#         v_emb = self.v_emb(v)
#         q_emb = self.q_emb(q)
        
#         # q_emb = q_emb[:, 0, :]
#         # v_emb = v_emb[:, 0, :]
        
#         # q_emb = q_emb.mean(1, keepdim =True)
#         # v_emb = v_emb.mean(1, keepdim =True)
#         # v_emb = v_emb.repeat_interleave(self.args.question_len, 1)
        
#         for co_att_layer in self.co_att_layers:
#             v_emb, q_emb = co_att_layer(v_emb, q_emb)
        
#         if self.fusion:
#             out = self.fusion(v_emb, q_emb)
#         else:
#             v_emb = v_emb.mean(1, keepdim =True)
#             v_emb = v_emb.repeat_interleave(self.args.question_len, 1)
            
#             out = q_emb * v_emb
        
#         out = out.mean(1, keepdim =True)
#         out = self.flatten(out)
        
#         # out = out.permute((0, 2, 1))
#         # out = out.mean(dim=-1)
        
#         return out
    
#     def classify(self, x):
#         return self.classifier(x)
    
# class Fusion(nn.Module):
#     """ Crazy multi-modal fusion: negative squared difference minus ReLU'd sum
#     """
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, y):
#         # found through grad student descent ;)
#         return - (x - y)**2 + F.ReLU(x + y)

# def build_CrossAtt(args):
#     print('Use BERT as question embedding...')
#     q_dim = args.q_dim
#     q_emb = BertQuestionEmbedding(args.bert_pretrained, args.device, use_mhsa=True)
#     utils.set_parameters_requires_grad(q_emb, True)

#     print('Loading image feature extractor...')
#     v_dim = args.v_dim
#     if args.object_detection:
#         v_emb = ObjectDetectionModel(args.image_pretrained, args.threshold, args.question_len)
#         utils.set_parameters_requires_grad(v_emb, False)  # freeze Object Detection model
#     else:
#         v_emb = initialize_backbone_model(args.backbone, use_imagenet_pretrained=args.image_pretrained)[0]

#     coatt_layers = nn.ModuleList([])
#     for _ in range(args.n_coatt):
#         coatt_layers.append(CoTransformerBlock(v_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout))

#     fusion = None
#     if args.object_detection:
#         fusion = FusionAttentionFeature(args)

#     classifier = SimpleClassifier(
#         args.joint_dim, args.joint_dim * 2, args.num_classes, args)

#     return CrossAttentionModel(q_emb, v_emb, coatt_layers, fusion, classifier, args)


# class Attention(nn.Module):
#     def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
#         super(Attention, self).__init__()
#         self.v_conv = nn.Conv2d(v_features, mid_features, 1, bias=False)  # let self.lin take care of bias
#         self.q_lin = nn.Linear(q_features, mid_features)
#         self.x_conv = nn.Conv2d(mid_features, glimpses, 1)

#         self.drop = nn.Dropout(drop)
#         self.relu = nn.ReLU(inplace=True)
#         self.fusion = Fusion()

#     def forward(self, v, q):
#         q_in = q
#         v = self.v_conv(self.drop(v))
#         q = self.q_lin(self.drop(q))
#         q = tile_2d_over_nd(q, v)
#         x = self.fusion(v, q)
#         x = self.x_conv(self.drop(x))
#         return x


# class GuidedAttentionModel(nn.Module):
#     def __init__(self, q_emb, v_embs, visual_guided_atts, visual_reduces, fusion, q_guided_att, classifier, args):
#         super(GuidedAttentionModel, self).__init__()
#         self.objects = args.objects
#         self.q_emb = q_emb
        
#         self.v_embs = nn.ModuleList(v_embs)
#         self.visual_guided_atts = nn.ModuleList(visual_guided_atts)
#         self.visual_reduces = nn.ModuleList(visual_reduces)
        
#         self.fusion = fusion
#         self.q_guided_att = q_guided_att
#         self.question_reduced = AttentionReduce(768, 768 // 2, 1)

#         self.classifier = classifier
#         self.flatten = nn.Flatten()
#         self.counter = Counter(self.objects)
#         self.last_linear = nn.Linear(args.joint_dim, args.joint_dim//2)
#         self.attention = Attention(
#             v_features=args.v_vit_dim,
#             q_features=args.q_dim,
#             mid_features=512,
#             glimpses=args.glimpse,
#             drop=0.5,
#         )
#         self.SiLU = torch.nn.SiLU()
    
#     def forward(self, v, q):
#         q_feat = self.q_emb(q)

#         v_feats = []
#         v_feat_saver = []
#         for v_emb, visual_guided_att, visual_reduce in zip(self.v_embs, self.visual_guided_atts, self.visual_reduces):
#             v_embed = v_emb(v)
#             v_feat_saver.append(v_embed)
#             v_guided = visual_guided_att(v_embed, q_feat)
#             # print("visual_reduce(v_guided, v_embed).shape:",visual_reduce(v_guided, v_embed).shape)
#             v_feats.append(visual_reduce(v_guided, v_embed))
#             # v_feats.append(visual_reduce(v_embed, v_embed))
#             # v_feats.append(v_guided.mean(1, keepdim=True))
        
#         v_feat_vit = v_feat_saver[0]
#         v_feat_cnn = v_feat_saver[1]
#         # this is where the counting component is used
#         # pick out the first attention map
#         # a = self.attention(v_feats[0],q_feat)

#         # a1 = a[:, 0, :, :].contiguous().view(a.size(0), -1)
#         # give it and the bounding boxes to the component
#         # self.counter(b, a1)
#         # v_joint_feat = self.fusion(*v_feats)
        
#         # v_joint_feat = torch.mul(*v_feats)
#         # v_joint_feat = torch.stack(v_feats, dim=-1).sum(-1)
#         v_joint_feat = torch.cat(v_feats, dim=1)
#         v_joint_feat = v_joint_feat.unsqueeze(1)
#         # print(v_joint_feat.shape)
#         # exit()
        
#         # out = out.mean(1, keepdim =True) # average pooling
#         # out = self.flatten(out)

#         # v_joint_feat = torch.cat(v_feats, dim=1)
#         # v_joint_feat = v_joint_feat.unsqueeze(1)

#         q_feat = self.q_guided_att(q_feat, v_joint_feat)
#         q_feat = q_feat.mean(1)
#         # out = self.question_reduced(q_feat, q_feat)
        
#         out = self.fusion(q_feat, v_joint_feat.squeeze(1))
#         out = self.SiLU(self.last_linear(out))
#         #out = self.classifier(out)
#         return out, v_feat_vit, v_feat_cnn
    
#     def classify(self, x):
#         return self.classifier(x)


# def build_GuidedAtt(args):
#     print('Use BERT as question embedding...')
#     q_dim = args.q_dim
#     q_emb = BertQuestionEmbedding(args.bert_pretrained, args.device, use_mhsa=True)
#     utils.set_parameters_requires_grad(q_emb, True)

#     print('Loading Vision Transformer feature extractor...')
#     v_vit_dim = args.v_vit_dim
#     v_vit_emb = initialize_backbone_model(args.vit_backbone, use_imagenet_pretrained=args.vit_image_pretrained)[0]

#     print(f'Loading CNN ({args.cnn_image_pretrained}) feature extractor...')
#     v_cnn_dim = args.v_cnn_dim
#     v_cnn_emb = initialize_backbone_model(args.cnn_image_pretrained, use_imagenet_pretrained=True)[0]
    
#     # args.v_common_dim = v_common_dim = v_vit_dim
#     cnn_converter = nn.Linear(v_cnn_dim, v_vit_dim)
#     v_cnn_emb = nn.Sequential(v_cnn_emb, cnn_converter)
#     v_cnn_dim = v_vit_dim
    
#     visual_vit_guided_att = GuidedTransformerEncoder(v_vit_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)
#     visual_cnn_guided_att = GuidedTransformerEncoder(v_cnn_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)

#     visual_vit_reduced = AttentionReduce(v_vit_dim, v_vit_dim // 2, args.glimpse)
#     visual_cnn_reduced = AttentionReduce(v_cnn_dim, v_cnn_dim // 2, args.glimpse)    
    
#     # question_guided_att = GuidedTransformerEncoder(q_dim, 1024, args.num_heads, args.hidden_dim, args.dropout)
#     question_guided_att = GuidedTransformerEncoder(q_dim, v_vit_dim + v_cnn_dim, args.num_heads, args.hidden_dim, args.dropout)

#     fusion = FusionLinear(q_dim, v_vit_dim + v_cnn_dim, args.joint_dim)

#     classifier = SimpleClassifier(
#         args.joint_dim//2, args.joint_dim , args.num_classes, args)

#     return GuidedAttentionModel(
#         q_emb, 
#         [v_vit_emb, v_cnn_emb], 
#         [visual_vit_guided_att, visual_cnn_guided_att],
#         [visual_vit_reduced, visual_cnn_reduced],
#         fusion,
#         question_guided_att,
#         classifier,
#         args
#     )
    
# def build_GuidedAtt_replaceResNetViT(args, replacingModel, replacingModelViT):
#     print('Use BERT as question embedding...')
#     q_dim = args.q_dim
#     q_emb = BertQuestionEmbedding(args.bert_pretrained, args.device, use_mhsa=True)
#     utils.set_parameters_requires_grad(q_emb, False) 

#     ########## Replace this ############
#     print('Loading Vision Transformer feature extractor...')
#     v_vit_dim = args.v_vit_dim
#     v_vit_emb = replacingModelViT
#     ########## Replace this ############

#     ########## Replace this ############
#     print(f'Loading CNN ({args.cnn_image_pretrained}) feature extractor...')
#     v_cnn_dim = args.v_cnn_dim
#     v_cnn_emb = replacingModel
#      ########## Replace this ############
#     # args.v_common_dim = v_common_dim = v_vit_dim
#     # print("v_cnn_dim:",v_cnn_dim)
#     # print("v_vit_dim:",v_vit_dim)
#     # exit()
#     cnn_converter = nn.Linear(v_cnn_dim, v_vit_dim)
#     v_cnn_emb = nn.Sequential(v_cnn_emb, cnn_converter)
#     v_cnn_dim = v_vit_dim
    
#     visual_vit_guided_att = GuidedTransformerEncoder(v_vit_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)
#     visual_cnn_guided_att = GuidedTransformerEncoder(v_cnn_dim, q_dim, args.num_heads, args.hidden_dim, args.dropout)

#     visual_vit_reduced = AttentionReduce(v_vit_dim, v_vit_dim // 2, args.glimpse)
#     visual_cnn_reduced = AttentionReduce(v_cnn_dim, v_cnn_dim // 2, args.glimpse)    
    
#     # question_guided_att = GuidedTransformerEncoder(q_dim, 1024, args.num_heads, args.hidden_dim, args.dropout)
#     question_guided_att = GuidedTransformerEncoder(q_dim, v_vit_dim + v_cnn_dim, args.num_heads, args.hidden_dim, args.dropout)

#     fusion = FusionLinear(q_dim, v_vit_dim + v_cnn_dim, args.joint_dim)

#     classifier = SimpleClassifier(
#         args.joint_dim//2, args.joint_dim , args.num_classes, args)

#     # classifier = Classifier(
#     #     in_features=(args.joint_dim,q_dim),
#     #     mid_features=args.joint_dim * 2,
#     #     out_features=args.num_classes,
#     #     count_features=args.objects + 1,
#     #     drop=args.dropout,
#     # )

#     return GuidedAttentionModel(
#         q_emb, 
#         [v_vit_emb, v_cnn_emb], 
#         [visual_vit_guided_att, visual_cnn_guided_att],
#         [visual_vit_reduced, visual_cnn_reduced],
#         fusion,
#         question_guided_att,
#         classifier,
#         args
#     )

class ColorGuidedAttentionModel(nn.Module):
    def __init__(self, q_emb, v_embs, visual_guided_atts, visual_reduces, fusion, q_guided_att, classifier, args):
        super(ColorGuidedAttentionModel, self).__init__()
        self.objects = 10
        self.q_emb = q_emb
        
        self.v_embs = nn.ModuleList(v_embs)
        self.visual_guided_atts = nn.ModuleList(visual_guided_atts)
        self.visual_reduces = nn.ModuleList(visual_reduces)
        
        self.fusion = fusion
        self.q_guided_att = q_guided_att

        self.classifier = classifier
        # self.flatten = nn.Flatten()
        # self.counter = Counter(self.objects)
        # self.last_linear = nn.Linear(args.joint_dim, args.joint_dim//2)
        # self.attention = Attention(
        #     v_features=args.v_vit_dim,
        #     q_features=args.q_dim,
        #     mid_features=512,
        #     glimpses=args.glimpse,
        #     drop=0.5,
        # )
        # self.SiLU = torch.nn.SiLU()
    
    def forward(self, v, q):
        q_feat = self.q_emb(q)

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
        q_feat = q_feat.mean(1)
        
        out = self.fusion(q_feat, v_joint_feat.squeeze(1))
        # out = self.SiLU(self.last_linear(out))
        #out = self.classifier(out)
        return out, v_feat_vit, v_feat_cnn
    
    def classify(self, x):
        return self.classifier(x)


def build_ColorGuidedAtt_replaceResNetViT(args, replacingModel, replacingModelViT):
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
        args.joint_dim, args.joint_dim*2 , args.num_classes, args)


    return ColorGuidedAttentionModel(
        q_emb, 
        [v_vit_emb, v_cnn_emb], 
        [visual_vit_guided_att, visual_cnn_guided_att],
        [visual_vit_reduced, visual_cnn_reduced],
        fusion,
        question_guided_att,
        classifier,
        args
    )