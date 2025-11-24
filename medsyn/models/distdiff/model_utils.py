'''
Copyright (c) Haowei Zhu, 2024
'''

import os
import os.path
import logging

import timm
import open_clip
from torch import nn
import torch
from dataloader import CUSTOM_TEMPLATES

logger = logging.getLogger(__name__)


def wrap_clip_forward(clip_model, text_feature):
    def custom_forward(self, x):
        x = self.encode_image(x)
        x = self.fc(x)
        return x

    fc = nn.Linear(text_feature.shape[1], text_feature.shape[0])
    with torch.no_grad():
        fc.weight.copy_(text_feature)
        nn.init.constant_(fc.bias, 0)
    clip_model.fc = fc
    clip_model.forward = custom_forward.__get__(clip_model)

    return clip_model

def add_encoder_image_method(model):
    def encoder_image_func(self, x, pooling='avg'):
        features = self.forward_features(x)
        if pooling == 'avg':
            features = nn.AdaptiveAvgPool2d((1, 1))(features)
        elif pooling == 'max':
            features = nn.AdaptiveMaxPool2d((1, 1))(features)
        else:
            raise ValueError("Unsupported pooling type. Please use 'avg' or 'max'.")
        features = torch.flatten(features, 1)
        return features
    model.encode_image = encoder_image_func.__get__(model)
    return model

def create_model(model_name, num_classes=1000, pretrained=False,
                 class_names=None, cache_dir=None, dataset_name=None, weight_path=None):

    print("=> creating model '{}'".format(model_name))
    if model_name == "resnet50":
        model = timm.create_model('resnet50')
        if pretrained:
            try:
                model = timm.create_model('resnet50', pretrained=True)
            except:
                model.load_state_dict(torch.load(f"save/{model_name}_imagenet1k.pth"))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = add_encoder_image_method(model)
    elif model_name == "resnext50":
        model = timm.create_model('resnext50_32x4d')
        if pretrained:
            model.load_state_dict(torch.load(f"save/{model_name}_imagenet1k.pth"))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = add_encoder_image_method(model)
    elif model_name == "mobilenetv2":
        model = timm.create_model('mobilenetv2_100')
        if pretrained:
            model.load_state_dict(torch.load(f"save/{model_name}_imagenet1k.pth"))
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        model = add_encoder_image_method(model)
    elif model_name == "wideresnet50":
        model = timm.create_model('wide_resnet50_2')
        if pretrained:
            model.load_state_dict(torch.load(f"save/{model_name}_imagenet1k.pth"))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = add_encoder_image_method(model)
    elif model_name == "open_clip_vit_b32":
        text_descriptions = [CUSTOM_TEMPLATES[dataset_name].format(label) for label in class_names]
        # Use LAION pretrained weights when pretrained=True
        # 'laion2b_s34b_b79k' is the standard pretrained version for ViT-B-32
        pretrained_version = 'laion2b_s34b_b79k' if pretrained else None
        print(f"=> OpenCLIP ViT-B-32: pretrained={pretrained}, using weights: {pretrained_version or 'random init'}")
        
        # Handle offline mode with cache_dir
        if cache_dir and pretrained:
            # Set HuggingFace cache environment variables
            os.environ['HF_HOME'] = cache_dir
            os.environ['TRANSFORMERS_CACHE'] = cache_dir
            
            # Check if weights exist in cache
            model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
            expected_cache_path = os.path.join(
                cache_dir, 
                f"models--{model_id.replace('/', '--')}",
                "snapshots"
            )
            
            if not os.path.exists(expected_cache_path):
                logger.warning(
                    f"Pretrained weights not found in cache: {expected_cache_path}\n"
                    f"Please download weights using:\n"
                    f"  python scripts/predownload_distdiff_models.py --cache_dir {cache_dir}"
                )
            else:
                print(f"Using cached weights from: {expected_cache_path}")
                
                # Force offline mode for HuggingFace Hub
                os.environ['HF_HUB_OFFLINE'] = '1'
        
        try:
            # Create model with OpenCLIP
            # open_clip.create_model_and_transforms handles HF_HOME automatically
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', 
                pretrained=pretrained_version, 
                cache_dir=cache_dir
            )
            
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            text_tokens = tokenizer(text_descriptions)
            text_features = model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            model = wrap_clip_forward(model, text_features)
            
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            if cache_dir and "offline mode" in str(e).lower():
                logger.error(
                    f"\n{'='*80}\n"
                    f"OFFLINE MODE ERROR\n"
                    f"{'='*80}\n"
                    f"Cache directory: {cache_dir}\n"
                    f"Model: {model_id}\n"
                    f"\nThe model weights are not available in the cache.\n"
                    f"To fix this, run on a machine with internet:\n"
                    f"  python scripts/predownload_distdiff_models.py \\\n"
                    f"    --cache_dir {cache_dir}\n"
                    f"{'='*80}"
                )
            raise
    else:
        raise NotImplementedError

    if weight_path is not None and weight_path != "None":
        weight = torch.load(weight_path)['state_dict']
        try:
            model.load_state_dict(weight, strict=True)
        except:
            new_state_dict = {}
            for key, value in weight.items():
                if key.startswith('module.'):
                    new_key = key[len('module.'):]
                else:
                    new_key = key
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=True)
        print(f"Load pretrained weights from : {weight_path}")

    return model


# model = create_model("open_clip_vit_b32", pretrained=True, class_names=["dog", "cat"])
# for name in ["resnet50", "resnext50", "mobilenetv2", "wideresnet50"]:
#     model = create_model(name, 100, pretrained=True, class_names=["dog", "cat"])
# output = model.encode_image(torch.randn([1, 3, 224, 224]))
# output = model.forward_features(torch.randn([1, 3, 224, 224]))
# print(output.shape)
