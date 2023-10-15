from constants import (
    PROJECT_DIRECTORY_PATH
)
import utils

import os
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
        self.device = device
        self.filepath = os.path.join(PROJECT_DIRECTORY_PATH, "data", "cbow", "model.pt")
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
        self.to(self.device)

    def save(self):
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        self.load_state_dict(torch.load(self.filepath, map_location=self.device))

    def get_embeddings(self) -> np.ndarray:
        embeddings = self.input_embeddings.weight.cpu().detach().numpy()
        embeddings = utils.normalize(embeddings, axis=1, keepdims=True)
        return embeddings

    def forward(self, context_words_idx):
        context_vector = torch.sum(self.input_embeddings(context_words_idx), dim=1)
        output = torch.matmul(context_vector, self.output_embeddings.weight.t())
        return torch.nn.functional.log_softmax(output, dim=1)
    
    def validate(self, validation_dataloader: datahandler.loaders.ValidationLoader) -> float:
        validation_dataloader.evaluate_analogies(self.get_embeddings())
        return validation_dataloader.analogies_accuracy()
