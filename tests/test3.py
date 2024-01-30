# Random input data for testing
import numpy as np
from whalegrad.nn.layers.transformer import Transformer



src = np.random.randn(32, 10, 512)  # Batch size: 32, Sequence length: 10, Model dimensions: 512
tgt = np.random.randn(32, 8, 512)   # Batch size: 32, Sequence length: 8, Model dimensions: 512
src_mask = np.random.randint(2, size=(32, 10, 10))  # Random binary mask for source sequence
tgt_mask = np.random.randint(2, size=(32, 8, 8))     # Random binary mask for target sequence
memory_mask = np.random.randint(2, size=(32, 10, 10))  # Random binary mask for memory

# Create the Transformer model
model = Transformer(dims=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6)

# Run the input through the model
output = model(src, tgt, src_mask, tgt_mask, memory_mask)

# Print the output shape
print("Output shape:", output.shape)
