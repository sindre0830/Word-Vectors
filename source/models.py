import utils

import torch
import numpy as np


class ModelCBOW(torch.nn.Module):
    def __init__(
            self,
            device: str,
            vocabulary_size: int,
            embedding_size: int,
            padding_idx: int = None
        ):
        super().__init__()
        # init embedding layers
        self.input_embeddings = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
            dtype=torch.float32,
            sparse=True
        )
        self.output_embeddings = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=padding_idx,
            dtype=torch.float32,
            sparse=True
        )
        # set the initial weights to be between -0.5 and 0.5
        self.input_embeddings.weight.data.uniform_(-0.5, 0.5)
        self.output_embeddings.weight.data.uniform_(-0.5, 0.5)
        # send model to device
        self.device = device
        self.to(self.device)

    def get_embeddings(self) -> np.ndarray:
        embeddings = self.input_embeddings.weight.cpu().detach().numpy()
        embeddings = utils.normalize(embeddings, axis=1, keepdims=True)
        return embeddings
