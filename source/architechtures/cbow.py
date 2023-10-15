from constants import (
    PROJECT_DIRECTORY_PATH
)
import utils
import datahandler.loaders
import models

import os
import numpy as np
import torch


def run() -> None:
    config = utils.load_config(filepath=os.path.join(PROJECT_DIRECTORY_PATH, "source", "config_cbow.yml"))
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # get corpus
    corpus = datahandler.loaders.Corpus()
    corpus.build(
        pipeline=[
            corpus.download,
            corpus.flatten
        ],
        data_directory="cbow"
    )
    # get vocabulary
    vocabulary = datahandler.loaders.Vocabulary(add_padding=True, add_unknown=False)
    vocabulary.build(corpus.words, config.vocabulary_size)
    # get training data
    training_dataloader = datahandler.loaders.DataLoaderCBOW(config.batch_size)
    training_dataloader.build(corpus.sentences, vocabulary, config.window_size, config.device)
    # get validation data
    validation_dataloader = datahandler.loaders.ValidationLoader(data_directory="cbow")
    validation_dataloader.build(vocabulary)
    # fit model and get embeddings
    model = models.ModelCBOW(config.device, config.vocabulary_size, config.embedding_size, vocabulary.padding_index)
    model.fit(
        training_dataloader,
        validation_dataloader,
        config.learning_rate,
        config.max_epochs,
        config.min_loss_improvement,
        config.patience,
        config.validation_interval
    )
    embeddings = model.get_embeddings()
    # evaluate embeddings
    validation_dataloader.evaluate_analogies(embeddings)
    validation_dataloader.evaluate_word_pair_similarity(embeddings)
    utils.print_divider()
    validation_dataloader.plot_analogies_rank(k=20)
    validation_dataloader.plot_word_pair_similarity()
    print(f"Analogy accuracy: {(validation_dataloader.analogies_accuracy() * 100):.2f}%")
    print(f"Spearman correlation coefficient: {validation_dataloader.word_pair_spearman_correlation():.5f}")
