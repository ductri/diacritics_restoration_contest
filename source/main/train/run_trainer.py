import logging

import torch

from data_for_train import dataset as my_dataset
from model_def.transformer.model_2 import Model
from model_def.transformer.training_function import TrainingFunction
from utils import pytorch_utils
from train.trainer import train


def input2_text(first_input, *params):
    return my_dataset.voc_src.idx2docs(first_input)


def target2_text(first_input, *params):
    return my_dataset.voc_tgt.idx2docs(first_input)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    NUM_WORKERS = 2
    PRINT_EVERY = 100
    PREDICT_EVERY = 50000
    EVAL_EVERY = 10000
    PRE_TRAINED_MODEL = '/source/main/train/output/saved_models/Model/2.1/70000.pt'

    my_dataset.bootstrap()
    train_loader = my_dataset.get_dl_train(batch_size=BATCH_SIZE, size=None)
    eval_loader = my_dataset.get_dl_eval(batch_size=BATCH_SIZE, size=None)
    logging.info('There will be %s steps for training, %s steps/epoch', NUM_EPOCHS * len(train_loader), len(train_loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    model.to(device)
    # logging.info('Model architecture: \n%s', model)
    logging.info('Total trainable parameters: %s', pytorch_utils.count_parameters(model))
    model = TrainingFunction(model)

    init_step = 0
    # Restore model
    if PRE_TRAINED_MODEL != '':
        checkpoint = torch.load(PRE_TRAINED_MODEL, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        init_step = checkpoint.get('step', 0)
        logging.info('Optimizer: %s', model.optimizer)
        logging.info('Load pre-trained model from %s successfully', PRE_TRAINED_MODEL)

    train(model, train_loader, eval_loader, dir_checkpoint='/source/main/train/output/', device=device,
          num_epoch=NUM_EPOCHS, print_every=PRINT_EVERY, predict_every=PREDICT_EVERY, eval_every=EVAL_EVERY,
          input_transform=input2_text, output_transform=target2_text, init_step=init_step, exp_id='2.1')
