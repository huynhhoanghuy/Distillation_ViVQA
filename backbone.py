from timm import create_model
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import utils_VQA as utils
from transformers import AutoFeatureExtractor, AutoModel, AutoConfig, DeiTModel,\
                         DetrForObjectDetection, YolosForObjectDetection

class CustomDenseNet121(models.densenet.DenseNet):
    def __init__(self, **kwargs):
        super(CustomDenseNet121, self).__init__(32, (6, 12, 24, 16), 64, **kwargs)
        # self.classifier = nn.nn.Identity()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = torch.flatten(out, 1)
        # out = self.classifier(out)
        return out


class CustomCNNModel(nn.Module):
    def __init__(self, model):
        super(CustomCNNModel, self).__init__()
        self.model = model

    def forward(self, x):
        x = x['pixel_values']
        out = self.model(x)[0]  # [batch, dim, w, h]
        b, dim, w, h = out.shape
        return out.view((b, dim, -1)).permute((0, 2, 1))


def load_pretrained_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model


class VisionTransformerModel(nn.Module):
    def __init__(self, pretrained):
        """Module for image embedding using pretrained Vision Transformers variants
        """
        super(VisionTransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrained)
        self.model = AutoModel.from_pretrained(pretrained)
        
    def forward(self, features):  
        output = self.model(**features)
        return output.last_hidden_state


class ObjectDetectionModel(nn.Module):
    def __init__(self, pretrained, threshold=0.7, max_objects=20):
        """Module for image embedding extracted from object detection model
        """
        super(ObjectDetectionModel, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrained)
        self.model = YolosForObjectDetection.from_pretrained(pretrained)
        self.threshold = threshold
        self.max_objects = max_objects
        
    def forward(self, features):  
        outputs = self.model(**features)
        return self.post_process_output(outputs)
    
    def post_process_output(self, outputs):
        b, k_obj, v_dim = outputs.last_hidden_state.shape  # (batch size, k objects, v_dim)
        
        # Calculate softmax logits
        probs = F.softmax(outputs.logits, dim=-1)[:,:,:-1]  # drop the last background class
        max_probs = probs.max(dim=-1).values   # get the classes with max probability
        
        sorted_probs, indices = torch.sort(max_probs, dim=-1, descending=True)
        
        # Do fancy index selection to sort the features by logits descending
        i = torch.arange(b).reshape(b, 1, 1) # shape = [b, 1, 1]
        j = indices.reshape(b, k_obj, 1)      # shape = [b, k object, 1]
        k = torch.arange(v_dim)                    # shape = [v_dim, ]

        outputs = outputs.last_hidden_state[i, j, k]  
        
        # Create mask to keep the classes with prob > threshold
        mask = (sorted_probs >= self.threshold).unsqueeze(-1)
        outputs = outputs * mask
        
        return outputs[:, 0:self.max_objects, :]  # keep the top n objects of prob


def initialize_backbone_model(model_name, is_training=True, use_imagenet_pretrained=True, model_path=None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 224 # default input_size for mostly models
    num_ftrs = 0
    channels = []
    layers = {}
    def get_interlayer(name):
        def hook(model, input, output):
            layers[name] = output.detach()
        return hook

    if "resnet" in model_name or 'vgg' in model_name:
        """ ResNet
        """
        model = create_model(model_name, pretrained=use_imagenet_pretrained, 
                             num_classes=0, features_only=True, out_indices=[-1])
        model_ft = CustomCNNModel(model)
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_imagenet_pretrained)
        # model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        # model_ft.num_classes = num_classes

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = CustomDenseNet121()
        model_ft.load_state_dict(models.densenet121(pretrained=use_imagenet_pretrained).state_dict())
        num_ftrs = model_ft.classifier.in_features
        # model_ft = nn.Sequential(*list(model_ft.children())[:-1]) # drop last layer
        # model_ft.classifier = nn.Identity()  # delete classifier layer
        model_ft.features.denseblock3.register_forward_hook(get_interlayer('layer3'))
        model_ft.features.denseblock2.register_forward_hook(get_interlayer('layer2'))
        model_ft.features.denseblock1.register_forward_hook(get_interlayer('layer1'))
        channels = [128, 256, 512, 1024]
        num_ftrs = num_ftrs * 7 * 7

    elif model_name == "efficientnet":
        """ EfficientNet
        """
        model_ft = models.efficientnet_b0(pretrained=use_imagenet_pretrained)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft = nn.Sequential(*list(model_ft.children())[:-2]) # drop 2 last layers
        model_ft[0][5].register_forward_hook(get_interlayer('layer3'))
        model_ft[0][3].register_forward_hook(get_interlayer('layer2'))
        model_ft[0][2].register_forward_hook(get_interlayer('layer1'))
        channels = [16, 24, 40, 112]
        num_ftrs = num_ftrs * 7 * 7

    elif model_name == 'convnext':
        model_ft = models.convnext_tiny(pretrained=use_imagenet_pretrained)
        # num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Identity()
        model_ft.avgpool = nn.Identity()
        model_ft.features[5].register_forward_hook(get_interlayer('layer3'))
        model_ft.features[3].register_forward_hook(get_interlayer('layer2'))
        model_ft.features[1].register_forward_hook(get_interlayer('layer1'))
        channels = [48, 96, 192, 384]
        num_ftrs = 768 * 7 * 7

    elif model_name == 'vit' or model_name == 'deit':
        
        model_ft = VisionTransformerModel(pretrained=use_imagenet_pretrained)
        # num_ftrs = model_ft.classifier.in_features
        # channels = [48, 96, 192, 384]
        # num_ftrs = 768 * 7 * 7
    elif model_name == 'detr':
        model_ft = ObjectDetectionModel(pretrained=use_imagenet_pretrained)

    else:
        print("Invalid model name, exiting...")
        exit()

    if model_path is not None:
        model_ft = load_pretrained_model(model_ft, model_path)
    
    utils.set_parameters_requires_grad(model_ft, is_training)

    return model_ft, num_ftrs, channels, layers
