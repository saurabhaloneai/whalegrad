import numpy as np
from whalegrad.nn.layers.embedding import Embedding  # Replace with the actual module or file where your Embedding class is defined

# # Create an instance of the Embedding layer
vocab_size = 10
embedding_dim = 5
embedding_layer = Embedding(vocab_size, embedding_dim)

# # Test case: create some input indices
input_indices = np.array([2, 5, 3, 8])

# # Forward pass through the embedding layer
embedding_output = embedding_layer(input_indices)

# # Verify the shape of the output
expected_shape = (len(input_indices), embedding_dim)
assert embedding_output.shape == expected_shape, f"Expected shape {expected_shape}, but got {embedding_output.shape}"

# # Print the input indices and corresponding embeddings for verification
for idx, embedding_vector in zip(input_indices, embedding_output):
    print(f"Input Index: {idx}, Embedding: {embedding_vector}")
# from whalegrad.nn.layers.positional_encoding import SinusoidalPositionalEncoding  # Replace with the actual module or file where your SinusoidalPositionalEncoding class is defined

# # Create an instance of the SinusoidalPositionalEncoding
# sequence_length = 10
# embedding_dim = 6
# positional_encoding = SinusoidalPositionalEncoding()

# # Test case: create some positions
# positions = np.arange(sequence_length)

# # Encode the positions using the SinusoidalPositionalEncoding
# encoded_positions = positional_encoding(positions)

# # Verify the shape of the output
# expected_shape = (sequence_length, embedding_dim)
# assert encoded_positions.shape == expected_shape, f"Expected shape {expected_shape}, but got {encoded_positions.shape}"

# # Print the positions and corresponding encoded vectors for verification
# for pos, encoding_vector in zip(positions, encoded_positions):
#     print(f"Position: {pos}, Encoding: {encoding_vector}")
# # Test code for SinusoidalPositionalEncoding with specified sequence_length and embedding_dim

# # Input parameters
# # sequence_length = 10
# # embedding_dim = 6
# # min_freq = 0.0001
# # max_freq = 1
# # scale = None
# # cos_first = False
# # full_turns = False

# # # Create an instance of SinusoidalPositionalEncoding
# # positional_encoder = SinusoidalPositionalEncoding(
# #     embedding_dim, min_freq, max_freq, scale, cos_first, full_turns
# # )

# # # Generate an array for testing
# # test_array = np.arange(sequence_length)

# # # Get the positional encoding
# # encoded_positions = positional_encoder(test_array)
# #  # Verify the shape of the output
# # expected_shape = (sequence_length, embedding_dim)
# # assert encoded_positions.shape == expected_shape, f"Expected shape {expected_shape}, but got {encoded_positions.shape}"

# # # Print the results
# # print("Input Array:")
# # print(test_array)

# # print("\nEncoded Positions Shape:")
# # print(encoded_positions.shape)  # Print the shape for debugging

# # # Add this line for debugging
# # # assert encoded_positions.shape == expected_shape, f"Expected shape {expected_shape}, but got {encoded_positions.shape}"
