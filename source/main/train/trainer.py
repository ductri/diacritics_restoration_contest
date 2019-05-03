import os
import time
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import torch

from utils import pytorch_utils, metrics
from utils.training_checker import TrainingChecker
from data_for_train import my_dataset
from model_def.baseline import Baseline


class MyTrainingChecker(TrainingChecker):
    def __init__(self, model, dir_checkpoint, init_score):
        super(MyTrainingChecker, self).__init__(model, dir_checkpoint + '/' + self._model.__class__.__name__, init_score)

    def save_model(self):
        file_name = os.path.join(self._dir_checkpoint, '%s.pt' % (self._step))
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'optimizer': self._model.optimizer.state_dict()}, file_name)


def cal_word_acc(prediction, target, seq_len):
    """
    All params are numpy arrays
    :param prediction: shape == (batch_size, max_len)
    :param target: shape == (batch_size, max_len)
    :param seq_len: shape == (batch_size)
    :return:
    """
    # shape == (batch_size, max_len)

    count_true = (prediction == target).astype(float)

    mask = np.zeros(count_true.shape)

    for idx, doc_len in enumerate(seq_len):
        mask[idx, :doc_len] = 1

    count_true *= mask

    acc = float(np.sum(count_true) / np.sum(seq_len))
    return acc


def cal_sen_acc(prediction, target, seq_len):
    """
    All params are numpy arrays
    :param prediction: shape == (batch_size, max_len)
    :param target: shape == (batch_size, max_len)
    :param seq_len: shape == (batch_size)
    :return:
    """
    # shape == (batch_size, max_len)
    mask = np.zeros_like(prediction)

    for idx, doc_len in enumerate(seq_len):
        mask[idx, :doc_len] = 1

    prediction = prediction * mask
    target = target * mask
    acc = np.sum(np.all(prediction == target, axis=1)) / prediction.shape[0]
    return acc


def train(model, train_loader, eval_loader, dir_checkpoint, device, num_epoch=10, print_every=1000, predict_every=500,
          eval_every=500, input_transform=None, output_transform=None):
    if input_transform is None:
        input_transform = lambda *x: x
    if output_transform is None:
        output_transform = lambda *x: x

    def predict_and_print_sample(input_tensors, target_tensor):
        sample_size = 3
        input_tensors = [input_tensor[:sample_size] for input_tensor in input_tensors]
        predict_tensor = model.cvt_output(model(*input_tensors))[:sample_size]
        target_tensor = target_tensor[:sample_size]

        input_transformed = input_transform(input_tensors[0].cpu().numpy())
        predict_transformed = output_transform(predict_tensor.cpu().numpy())
        target_transformed = output_transform(target_tensor.cpu().numpy())

        for idx, (src, pred, tgt) in enumerate(zip(input_transformed, predict_transformed, target_transformed)):
            logging.info('Sample %s ', idx + 1)
            logging.info('Source:\t%s', src)
            logging.info('Predict:\t%s', pred)
            logging.info('Target:\t%s', tgt)
            logging.info('------')

    t_loss_tracking = metrics.MeanMetrics()
    e_loss_tracking = metrics.MeanMetrics()
    e_w_a_tracking = metrics.MeanMetrics()
    e_s_a_tracking = metrics.MeanMetrics()
    training_checker = MyTrainingChecker(model, dir_checkpoint, init_score=0)

    step = 0
    model.to(device)
    logging.info('----------------------- START TRAINING -----------------------')
    for _ in range(num_epoch):
        for inputs in train_loader:
            inputs = [i.to(device) for i in inputs]
            start = time.time()
            train_loss = model.train_batch(*inputs)
            t_loss_tracking.add(train_loss)
            step += 1
            with torch.no_grad():
                if step % print_every == 0 or step == 1:
                    model.eval()
                    prediction_numpy = model.cvt_output(model(*inputs[:1])).cpu().numpy()
                    target_numpy = inputs[2].cpu().numpy()
                    seq_len_numpy = inputs[1].cpu().numpy()
                    w_acc = cal_word_acc(prediction_numpy, target_numpy, seq_len_numpy)
                    s_acc = cal_sen_acc(prediction_numpy, target_numpy, seq_len_numpy)

                    logging.info(
                        'Step: %s \t L_mean: %.5f \t L_std: %.5f \t w_a: %.4f \t s_a: %.4f \t Duration: %.3f s/step' % (
                            step, t_loss_tracking.mean(), np.std(t_loss_tracking.figures), w_acc, s_acc, time.time() - start))
                    t_loss_tracking.reset()

                if step % predict_every == 0:
                    model.eval()

                    logging.info('\n\n------------------ Predict samples from train ------------------ ')
                    logging.info('Step: %s', step)
                    predict_and_print_sample(inputs[:1], inputs[2])

                if step % eval_every == 0:
                    model.eval()
                    e_loss_tracking.reset()
                    e_w_a_tracking.reset()
                    e_s_a_tracking.reset()

                    start = time.time()
                    for eval_inputs in eval_loader:
                        eval_inputs = [i.to(device) for i in eval_inputs]
                        eval_pred_tensor = model(*eval_inputs[:1])
                        eval_loss = model.loss(eval_pred_tensor, eval_inputs[1], eval_inputs[2]).item()
                        e_loss_tracking.add(eval_loss)
                        eval_prediction = model.cvt_output(eval_pred_tensor)
                        e_pred_numpy = eval_prediction.cpu().numpy()
                        e_target_numpy = eval_inputs[2].cpu().numpy()
                        e_seq_len_numpy = eval_inputs[1].cpu().numpy()
                        e_w_a_tracking.add(cal_word_acc(e_pred_numpy, e_target_numpy, e_seq_len_numpy))
                        e_s_a_tracking.add(cal_sen_acc(e_pred_numpy, e_target_numpy, e_seq_len_numpy))
                    logging.info('\n\n------------------ \tEvaluation\t------------------')
                    logging.info('Step: %s', step)
                    logging.info('Number of batchs: %s', e_loss_tracking.get_count())
                    logging.info('L_mean: %.5f \t L_std: %.5f \t  w_a: %.5f \t s_a: %.5f \t Duration: %.3f s/step' %
                                 (e_loss_tracking.mean(), np.std(e_loss_tracking.figures), e_w_a_tracking.mean(),
                                  e_s_a_tracking.mean(), time.time() - start))

                    training_checker.update(e_w_a_tracking.mean(), step)
                    best_score, best_score_step = training_checker.best()
                    logging.info('Current best score: %s recorded at step %s', best_score, best_score_step)

                    eval_inputs = next(iter(eval_loader))
                    eval_inputs = [item.to(device) for item in eval_inputs]
                    predict_and_print_sample(eval_inputs[:1], eval_inputs[2])


def input2_text(*params):
    """

    :param input_tensor: shape == (batch_size, max_len)
    :return:
    """
    first_input = params[0]
    return my_dataset.voc_src.idx2docs(first_input)


def target2_text(*params):
    """

    :param target: shape == (batch_size, max_len, no_class)
    :param source: batch_size list of str
    :return:
    """
    first_input = params[0]
    return my_dataset.voc_tgt.idx2docs(first_input)


if __name__ == '__main__':
    BATCH_SIZE = 32
    NUM_EPOCHS = 500
    NUM_WORKERS = 0
    PRINT_EVERY = 200
    EVAL_EVERY = 5000
    PRE_TRAINED_MODEL = ''

    my_dataset.bootstrap()
    train_loader = my_dataset.get_dl_train(batch_size=BATCH_SIZE, size=100)
    eval_loader = my_dataset.get_dl_eval(batch_size=BATCH_SIZE)

    model = Baseline(src_word_vocab_size=len(my_dataset.voc_src.index2word),
                     tgt_word_vocab_size=len(my_dataset.voc_tgt.index2word))

    logging.info('Model architecture: \n%s', model)
    logging.info('Total trainable parameters: %s', pytorch_utils.count_parameters(model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.build_stuff_for_training(device)

    # Restore model
    if PRE_TRAINED_MODEL != '':
        checkpoint = torch.load(PRE_TRAINED_MODEL, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info('Load pre-trained model from %s successfully', PRE_TRAINED_MODEL)

    train(model, train_loader, eval_loader, dir_checkpoint='main/train/output/saved_models/', device=device,
          num_epoch=NUM_EPOCHS, print_every=PRINT_EVERY, predict_every=EVAL_EVERY, eval_every=EVAL_EVERY,
          input_transform=input2_text, output_transform=target2_text)
