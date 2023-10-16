from constants import (
    PROJECT_DIRECTORY_PATH
)
import utils
import datahandler.loaders

import os
import torch
import numpy as np
import tqdm


class ModelGloVe(torch.nn.Module):
    def __init__(
            self,
            device: str,
            vocabulary_size: int,
            embedding_size: int,
            x_max: float,
            alpha: float,
            padding_idx: int = None
        ):
        super().__init__()
        self.device = device
        self.filepath = os.path.join(PROJECT_DIRECTORY_PATH, "data", "cbow", "model.pt")
        self.x_max = x_max
        self.alpha = alpha
        self.padding_idx = padding_idx
        # init embedding layers
        self.main_embeddings = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=self.padding_idx,
            dtype=torch.float32,
            sparse=True
        )
        self.context_embeddings = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=self.padding_idx,
            dtype=torch.float32,
            sparse=True
        )
        self.main_bias = torch.nn.Parameter(torch.randn(vocabulary_size, dtype=torch.float32))
        self.context_bias = torch.nn.Parameter(torch.randn(vocabulary_size, dtype=torch.float32))
        # set the initial weights to be between -0.5 and 0.5
        self.main_embeddings.weight.data.uniform_(-0.5, 0.5)
        self.context_embeddings.weight.data.uniform_(-0.5, 0.5)
        # set padding vector to zero
        if self.padding_idx is not None:
            self.main_embeddings.weight.data[self.padding_idx, :] = 0
            self.context_embeddings.weight.data[self.padding_idx, :] = 0
        # send model to device
        self.to(self.device)

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        self.load_state_dict(torch.load(self.filepath, map_location=self.device))

    def get_embeddings(self) -> np.ndarray:
        embeddings: np.ndarray = (self.main_embeddings.weight.cpu().detach() + self.context_embeddings.weight.cpu().detach()).numpy()
        # if padding is used, set it to 1 to avoid division by zero
        if self.padding_idx is not None:
            embeddings[self.padding_idx] = 1
        embeddings = utils.normalize(embeddings, axis=1, keepdims=True)
        return embeddings

    def forward(self, word_index: torch.Tensor, context_index: torch.Tensor, cooccurrence_count: torch.Tensor):
        # dot product calculation
        dot_product: torch.Tensor = (self.main_embeddings(word_index) * self.context_embeddings(context_index)).sum(dim=1)
        # prediction
        prediction = dot_product + self.main_bias[word_index] + self.context_bias[context_index]
        # weighted loss calculation
        x_scaled = (cooccurrence_count / self.x_max).pow(self.alpha)
        weighted_error = (x_scaled.clamp(0, 1) * (prediction - cooccurrence_count.log()) ** 2).mean()
        return weighted_error
    
    def validate(self, validation_dataloader: datahandler.loaders.ValidationLoader) -> float:
        validation_dataloader.evaluate_analogies(self.get_embeddings(), quiet=True)
        return validation_dataloader.analogies_accuracy()


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
        self.padding_idx = padding_idx
        # init embedding layers
        self.input_embeddings = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=self.padding_idx,
            dtype=torch.float32,
            sparse=True
        )
        self.output_embeddings = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=self.padding_idx,
            dtype=torch.float32,
            sparse=True
        )
        # set the initial weights to be between -0.5 and 0.5
        self.input_embeddings.weight.data.uniform_(-0.5, 0.5)
        self.output_embeddings.weight.data.uniform_(-0.5, 0.5)
        # set padding vector to zero
        if self.padding_idx is not None:
            self.input_embeddings.weight.data[self.padding_idx, :] = 0
            self.output_embeddings.weight.data[self.padding_idx, :] = 0
        # send model to device
        self.to(self.device)

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        self.load_state_dict(torch.load(self.filepath, map_location=self.device))

    def get_embeddings(self) -> np.ndarray:
        embeddings: np.ndarray = self.input_embeddings.weight.cpu().detach().numpy()
        # if padding is used, set it to 1 to avoid division by zero
        if self.padding_idx is not None:
            embeddings[self.padding_idx] = 1
        embeddings = utils.normalize(embeddings, axis=1, keepdims=True)
        return embeddings

    def forward(self, context_words_idx):
        context_vector = torch.sum(self.input_embeddings(context_words_idx), dim=1)
        output = torch.matmul(context_vector, self.output_embeddings.weight.t())
        return torch.nn.functional.log_softmax(output, dim=1)
    
    def validate(self, validation_dataloader: datahandler.loaders.ValidationLoader) -> float:
        validation_dataloader.evaluate_analogies(self.get_embeddings(), quiet=True)
        return validation_dataloader.analogies_accuracy()

    def fit(
            self,
            training_dataloader: datahandler.loaders.DataLoaderCBOW,
            validation_dataloader: datahandler.loaders.ValidationLoader,
            learning_rate: float,
            max_epochs: int,
            min_loss_improvement: float,
            patience: int,
            validation_interval: int
        ):
        # check if cache exists
        if os.path.exists(self.filepath):
            progress_bar = tqdm.tqdm(desc="Loading cached model", total=1)
            self.load()
            progress_bar.update(1)
            return
        print("Training model:")
        loss_history = []
        acc_history = []
        dataset_size = len(training_dataloader)
        last_batch_index = dataset_size - 1
        # set optimizer and critirion
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = torch.nn.NLLLoss()
        # loop through each epoch
        best_loss = float("inf")
        best_acc = -float("inf")
        epochs_without_improvement = 0
        for epoch in range(max_epochs):
            total_loss = 0.0
            # define the progressbar
            progressbar = utils.get_model_progressbar(training_dataloader, epoch, max_epochs)
            # set model to training mode
            self.train()
            # loop through the dataset
            for idx, batch in enumerate(progressbar):
                # clear gradients
                optimizer.zero_grad()
                # unpack batch data and send to device
                context_words: torch.Tensor = batch[0]
                target_words: torch.Tensor = batch[1]
                # compute gradients
                outputs: torch.Tensor = self(context_words)
                loss: torch.Tensor = criterion(outputs, target_words)
                total_loss += loss.item()
                # apply gradients
                loss.backward()
                optimizer.step()
                # branch if on last iteration
                if idx == last_batch_index:
                    # update early stopping and save model
                    train_loss = total_loss / (idx + 1)
                    if train_loss <= (best_loss - min_loss_improvement):
                        best_loss = train_loss
                        epochs_without_improvement = 0
                        # save best model
                        self.save()
                    else:
                        epochs_without_improvement += 1
                    # validate model every n epochs
                    if epoch == 0 or (epoch + 1) == max_epochs or (epoch + 1) % validation_interval == 0:
                        self.eval()
                        train_acc = self.validate(validation_dataloader)
                        if train_acc >= best_acc:
                            best_acc = train_acc
                        self.train()
                    else:
                        train_acc = acc_history[-1]
                    # add to history and plot
                    loss_history.append(train_loss)
                    acc_history.append(train_acc)
                    utils.plot_loss_and_accuracy(loss_history, acc_history, data_directory="cbow")
                    # update information with current values
                    utils.set_model_progressbar_prefix(progressbar, train_loss, best_loss, train_acc, best_acc)
            # check for early stopping
            if epochs_without_improvement >= patience:
                break
        # load the best model from training
        self.load()
        # empty GPU cache
        if "cuda" in self.device:
            torch.cuda.empty_cache()


class ModelSkipGram(torch.nn.Module):
    def __init__(
            self,
            device: str,
            vocabulary_size: int,
            embedding_size: int,
            padding_idx: int = None
        ):
        super().__init__()
        self.device = device
        self.filepath = os.path.join(PROJECT_DIRECTORY_PATH, "data", "skipgram", "model.pt")
        self.padding_idx = padding_idx
        # init embedding layers
        self.input_embeddings = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=self.padding_idx,
            dtype=torch.float32,
            sparse=True
        )
        self.output_embeddings = torch.nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=self.padding_idx,
            dtype=torch.float32,
            sparse=True
        )
        # set the initial weights to be between -0.5 and 0.5
        self.input_embeddings.weight.data.uniform_(-0.5, 0.5)
        self.output_embeddings.weight.data.uniform_(-0.5, 0.5)
        # set padding vector to zero
        if self.padding_idx is not None:
            self.input_embeddings.weight.data[self.padding_idx, :] = 0
            self.output_embeddings.weight.data[self.padding_idx, :] = 0
        # send model to device
        self.to(self.device)

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        torch.save(self.state_dict(), self.filepath)

    def load(self):
        self.load_state_dict(torch.load(self.filepath, map_location=self.device))

    def get_embeddings(self) -> np.ndarray:
        embeddings: np.ndarray = self.input_embeddings.weight.cpu().detach().numpy()
        # if padding is used, set it to 1 to avoid division by zero
        if self.padding_idx is not None:
            embeddings[self.padding_idx] = 1
        embeddings = utils.normalize(embeddings, axis=1, keepdims=True)
        return embeddings

    def forward(self, target_words):
        target_vector = self.input_embeddings(target_words)
        output = torch.matmul(target_vector, self.output_embeddings.weight.t())
        return torch.nn.functional.log_softmax(output, dim=1)
    
    def validate(self, validation_dataloader: datahandler.loaders.ValidationLoader) -> float:
        validation_dataloader.evaluate_analogies(self.get_embeddings(), quiet=True)
        return validation_dataloader.analogies_accuracy()

    def fit(
            self,
            training_dataloader: datahandler.loaders.DataLoaderCBOW,
            validation_dataloader: datahandler.loaders.ValidationLoader,
            learning_rate: float,
            max_epochs: int,
            min_loss_improvement: float,
            patience: int,
            validation_interval: int
        ):
        # check if cache exists
        if os.path.exists(self.filepath):
            progress_bar = tqdm.tqdm(desc="Loading cached model", total=1)
            self.load()
            progress_bar.update(1)
            return
        print("Training model:")
        loss_history = []
        acc_history = []
        dataset_size = len(training_dataloader)
        last_batch_index = dataset_size - 1
        # set optimizer and critirion
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        criterion = torch.nn.NLLLoss()
        # loop through each epoch
        best_loss = float("inf")
        best_acc = -float("inf")
        epochs_without_improvement = 0
        for epoch in range(max_epochs):
            total_loss = 0.0
            # define the progressbar
            progressbar = utils.get_model_progressbar(training_dataloader, epoch, max_epochs)
            # set model to training mode
            self.train()
            # loop through the dataset
            for idx, batch in enumerate(progressbar):
                # clear gradients
                optimizer.zero_grad()
                # unpack batch data and send to device
                context_words: torch.Tensor = batch[0]
                target_words: torch.Tensor = batch[1]
                # compute gradients
                outputs: torch.Tensor = self(target_words)
                loss: torch.Tensor = criterion(outputs, context_words)
                total_loss += loss.item()
                # apply gradients
                loss.backward()
                optimizer.step()
                # branch if on last iteration
                if idx == last_batch_index:
                    # update early stopping and save model
                    train_loss = total_loss / (idx + 1)
                    if train_loss <= (best_loss - min_loss_improvement):
                        best_loss = train_loss
                        epochs_without_improvement = 0
                        # save best model
                        self.save()
                    else:
                        epochs_without_improvement += 1
                    # validate model every n epochs
                    if epoch == 0 or (epoch + 1) == max_epochs or (epoch + 1) % validation_interval == 0:
                        self.eval()
                        train_acc = self.validate(validation_dataloader)
                        if train_acc >= best_acc:
                            best_acc = train_acc
                        self.train()
                    else:
                        train_acc = acc_history[-1]
                    # add to history and plot
                    loss_history.append(train_loss)
                    acc_history.append(train_acc)
                    utils.plot_loss_and_accuracy(loss_history, acc_history, data_directory="skipgram")
                    # update information with current values
                    utils.set_model_progressbar_prefix(progressbar, train_loss, best_loss, train_acc, best_acc)
            # check for early stopping
            if epochs_without_improvement >= patience:
                break
        # load the best model from training
        self.load()
        # empty GPU cache
        if "cuda" in self.device:
            torch.cuda.empty_cache()
