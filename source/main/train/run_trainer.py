import logging
import torch

from data_for_train import target as my_dataset
from model_def.seq2seq import Seq2Seq
from utils import pytorch_utils
from train.trainer import train


def input2_text(first_input, *params):
    return my_dataset.docs.voc.idx2docs(first_input)


def target2_text(first_input, *params):
    return my_dataset.docs.voc.idx2docs(first_input)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    BATCH_SIZE = 32
    NUM_EPOCHS = 500
    NUM_WORKERS = 0
    PRINT_EVERY = 200
    PREDICT_EVERY = 5000
    EVAL_EVERY = 10000
    PRE_TRAINED_MODEL = ''

    my_dataset.bootstrap()
    train_loader = my_dataset.get_dl_train(batch_size=BATCH_SIZE, size=None)
    eval_loader = my_dataset.get_dl_eval(batch_size=BATCH_SIZE)

    model = Seq2Seq(src_vocab_size=len(my_dataset.docs.voc.index2word),
                    tgt_vocab_size=len(my_dataset.docs.voc.index2word),
                    start_idx=2,
                    end_idx=3
                    )
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info('Model architecture: \n%s', model)
    logging.info('Total trainable parameters: %s', pytorch_utils.count_parameters(model))

    init_step = 0
    # Restore model
    if PRE_TRAINED_MODEL != '':
        checkpoint = torch.load(PRE_TRAINED_MODEL, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        init_step = checkpoint.get('step', 0)
        logging.info('Load pre-trained model from %s successfully', PRE_TRAINED_MODEL)

    train(model, train_loader, eval_loader, dir_checkpoint='/source/main/train/output/saved_models/', device=device,
          num_epoch=NUM_EPOCHS, print_every=PRINT_EVERY, predict_every=PREDICT_EVERY, eval_every=EVAL_EVERY,
          input_transform=input2_text, output_transform=target2_text, init_step=init_step)
