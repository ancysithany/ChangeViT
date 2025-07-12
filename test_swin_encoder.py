import torch
from model.encoder import Encoder  # or SwinEncoder, depending on what you named it

# Dummy input
img1 = torch.randn(2, 3, 256, 256)  # Batch of 2
img2 = torch.randn(2, 3, 256, 256)

# Load encoder
encoder = Encoder(model_type='tiny')

# Forward pass
out1, out2 = encoder(img1, img2)

# Print output shapes
print("Image 1 Feature Shapes:")
for o in out1:
    print(o.shape)

print("\nImage 2 Feature Shapes:")
for o in out2:
    print(o.shape)