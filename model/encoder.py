from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from einops import rearrange
import timm

from model.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from model.resnet import resnet18


class SwinTransformerWrapper(nn.Module):
    """Wrapper to make Swin Transformer output compatible with original encoder format"""
    
    def __init__(self, model_name, img_size=256, pretrained=True, features_only=True):
        super().__init__()
        self.swin = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=features_only,
            img_size=img_size,
            out_indices=[3]  # Get the last feature map
        )
        
        # Get the feature dimensions from the model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            dummy_output = self.swin(dummy_input)
            self.feature_dim = dummy_output[0].shape[1]  # Channel dimension
            self.spatial_size = dummy_output[0].shape[2]  # Spatial dimension (H=W)
    
    def forward(self, x):
        # Extract features from Swin Transformer
        features = self.swin(x)
        # Return the last feature map (highest level features)
        return features[0]  # Shape: [B, C, H, W]


class Encoder(nn.Module):
    def __init__(self, model_type='small'):
        super().__init__()
        
        if model_type == 'tiny':
            # Use Swin Tiny instead of DINOv2 tiny
            self.swin_model = 'swin_tiny_patch4_window7_224'
            self.vit = SwinTransformerWrapper(
                model_name=self.swin_model,
                img_size=256,
                pretrained=True
            )
            print('Using Swin Tiny model')
            
        elif model_type == 'small':
            # Use Swin Small instead of DINOv2 small
            self.swin_model = 'swin_small_patch4_window7_224'
            self.vit = SwinTransformerWrapper(
                model_name=self.swin_model,
                img_size=256,
                pretrained=True
            )
            print('Using Swin Small model')
            
        elif model_type == 'base':
            # Use Swin Base 
            self.swin_model = 'swin_base_patch4_window7_224'
            self.vit = SwinTransformerWrapper(
                model_name=self.swin_model,
                img_size=256,
                pretrained=True
            )
            print('Using Swin Base model')
            
        else:
            assert False, r'Encoder: check the model type (tiny, small, base)'

        print(f'Model type: {model_type}, Swin model: {self.swin_model}')
        
        # Keep the ResNet for detail capture (unchanged)
        self.resnet = resnet18(pretrained=True)
        self.drop = nn.Dropout(p=0.01)

        # Add projection layer to match expected output dimensions if needed
        # This ensures compatibility with your decoder
        expected_dim = 384 if model_type == 'small' else 192 if model_type == 'tiny' else 768
        if self.vit.feature_dim != expected_dim:
            self.feature_projection = nn.Conv2d(self.vit.feature_dim, expected_dim, 1)
        else:
            self.feature_projection = nn.Identity()
            
        # Add spatial adjustment if needed to match 16x16 output
        if self.vit.spatial_size != 16:
            self.spatial_adjust = nn.AdaptiveAvgPool2d((16, 16))
        else:
            self.spatial_adjust = nn.Identity()

    def detail_capture(self, x):
        """ResNet detail capture - unchanged from original"""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x2 = self.drop(self.resnet.layer1(x))
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3(x3)

        return [x2, x3, x4]

    def forward(self, x, y):
        """
        Forward pass maintaining exact same output format as original
        Returns: c_x + [v_x], c_y + [v_y]
        where c_x, c_y are ResNet features and v_x, v_y are transformer features
        """
        # Get Swin Transformer features
        v_x = self.vit(x)  # [B, C, H, W]
        v_y = self.vit(y)  # [B, C, H, W]
        
        # Apply projection and spatial adjustment to match expected format
        v_x = self.feature_projection(v_x)
        v_y = self.feature_projection(v_y)
        
        v_x = self.spatial_adjust(v_x)  # Ensure 16x16 spatial size
        v_y = self.spatial_adjust(v_y)  # Ensure 16x16 spatial size

        # Get ResNet detail features (unchanged)
        c_x = self.detail_capture(x)
        c_y = self.detail_capture(y)

        # Return in exact same format as original: [resnet_features] + [transformer_features]
        return c_x + [v_x], c_y + [v_y]


# Alternative implementation if you want to use specific Swin models
class SwinEncoder(nn.Module):
    """Alternative encoder using specific Swin configurations"""
    
    def __init__(self, model_type='small'):
        super().__init__()
        
        # Swin model configurations
        swin_configs = {
            'tiny': {
                'model_name': 'swin_tiny_patch4_window7_224',
                'expected_dim': 192
            },
            'small': {
                'model_name': 'swin_small_patch4_window7_224', 
                'expected_dim': 384
            },
            'base': {
                'model_name': 'swin_base_patch4_window7_224',
                'expected_dim': 768
            }
        }
        
        if model_type not in swin_configs:
            raise ValueError(f"Model type {model_type} not supported. Use 'tiny', 'small', or 'base'")
            
        config = swin_configs[model_type]
        
        # Create Swin model
        self.swin = timm.create_model(
            config['model_name'],
            pretrained=True,
            features_only=True,
            img_size=256,
            out_indices=[3]  # Last stage features
        )
        
        # Get actual feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            dummy_output = self.swin(dummy_input)
            actual_dim = dummy_output[0].shape[1]
            spatial_size = dummy_output[0].shape[2]
        
        # Project to expected dimensions
        self.feature_projection = nn.Conv2d(actual_dim, config['expected_dim'], 1) if actual_dim != config['expected_dim'] else nn.Identity()
        
        # Ensure 16x16 output
        self.spatial_adjust = nn.AdaptiveAvgPool2d((16, 16)) if spatial_size != 16 else nn.Identity()
        
        # ResNet for detail capture
        self.resnet = resnet18(pretrained=True)
        self.drop = nn.Dropout(p=0.01)
        
        print(f'Model type: {model_type}')
        print(f'Swin model: {config["model_name"]}')
        print(f'Feature dim: {actual_dim} -> {config["expected_dim"]}')
        print(f'Spatial size: {spatial_size} -> 16')

    def detail_capture(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x2 = self.drop(self.resnet.layer1(x))
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3(x3)

        return [x2, x3, x4]

    def forward(self, x, y):
        # Swin features
        swin_x = self.swin(x)[0]  # Get last stage features
        swin_y = self.swin(y)[0]
        
        # Apply projections
        v_x = self.spatial_adjust(self.feature_projection(swin_x))
        v_y = self.spatial_adjust(self.feature_projection(swin_y))

        # ResNet features
        c_x = self.detail_capture(x)
        c_y = self.detail_capture(y)

        return c_x + [v_x], c_y + [v_y]


# For backwards compatibility, use the main Encoder class
# You can also use SwinEncoder if you prefer the alternative implementation
