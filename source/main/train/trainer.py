import os
import time
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import torch

from naruto_skills.dl_logging import DLLoggingHandler, DLTBHandler, DLLogger
from utils.training_checker import TrainingChecker
from model_def.baseline import Baseline
from model_def.simple_but_huge import SimpleButHuge


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
          eval_every=500, input_transform=None, output_transform=None, init_step=0):
    if input_transform is None:
        input_transform = lambda *x: x
    if output_transform is None:
        output_transform = lambda *x: x

    def predict_and_print_sample(*inputs):
        sample_size = 5
        input_tensors = [input_tensor[:sample_size] for input_tensor in inputs]
        predict_tensor = model(input_tensors[0])

        input_transformed = input_transform(input_tensors[0].cpu().numpy())
        target_transformed = output_transform(input_tensors[1].cpu().numpy())

        seq_len_np = input_tensors[2].cpu().numpy()
        predict_transformed = output_transform(predict_tensor.cpu().numpy())
        predict_transformed = [' '.join(doc.split()[:seq_len_np[idx]]) for idx, doc in enumerate(predict_transformed)]

        for idx, (src, pred, tgt) in enumerate(zip(input_transformed, predict_transformed, target_transformed)):
            logging.info('Sample %s ', idx + 1)
            logging.info('Source:\t%s', src)
            logging.info('Predict:\t%s', pred)
            logging.info('Target:\t%s', tgt)
            logging.info('------')

    training_checker = TrainingChecker(model, root_dir=dir_checkpoint+'/saved_models', init_score=0)
    my_logger = DLLogger()
    my_logger.add_handler(DLLoggingHandler())
    my_logger.add_handler(DLTBHandler(dir_checkpoint+'/logging/%s/%s' % (model.__class__.__name__, training_checker.exp_id)))

    e_w_a_tracking = []
    e_s_a_tracking = []

    t_loss_tracking_tag = 'train/loss'
    t_w_a_tracking_tag = 'train/word_acc'
    t_s_a_tracking_tag = 'train/sen_acc'
    e_w_a_mean_tag = 'eval/word_acc'
    e_s_a_mean_tag = 'eval/sen_acc'
    e_w_a_std_tag = 'eval/word_acc_std'
    e_s_a_sdt_tag = 'eval/sen_acc_std'
    train_duration_tag = 'train/step_duration'
    eval_duration_tag = 'eval/step_duration'

    step = init_step
    model.to(device)
    logging.info('----------------------- START TRAINING -----------------------')
    for _ in range(num_epoch):
        for inputs in train_loader:
            inputs = [i.to(device) for i in inputs]
            start = time.time()
            train_loss = model.train_batch(*inputs)
            # my_logger.add_scalar(t_loss_tracking_tag, train_loss, step)
            step += 1
            with torch.no_grad():
                if step % print_every == 0 or step == 1:
                    model.eval()
                    prediction_numpy = model(inputs[0]).cpu().numpy()
                    target_numpy = inputs[1].cpu().numpy()
                    seq_len_numpy = inputs[2].cpu().numpy()
                    w_acc = cal_word_acc(prediction_numpy, target_numpy, seq_len_numpy)
                    s_acc = cal_sen_acc(prediction_numpy, target_numpy, seq_len_numpy)

                    my_logger.add_scalar(t_loss_tracking_tag, train_loss, step)
                    my_logger.add_scalar(t_w_a_tracking_tag, w_acc, step)
                    my_logger.add_scalar(t_s_a_tracking_tag, s_acc, step)
                    my_logger.add_scalar(train_duration_tag, time.time() - start, step)

                if step % predict_every == 0:
                    model.eval()

                    logging.info('\n\n------------------ Predict samples from train ------------------ ')
                    logging.info('Step: %s', step)
                    predict_and_print_sample(*inputs)

                if step % eval_every == 0:
                    model.eval()
                    e_w_a_tracking.clear()
                    e_s_a_tracking.clear()

                    start = time.time()
                    for eval_inputs in eval_loader:
                        eval_inputs = [i.to(device) for i in eval_inputs]
                        e_pred_numpy = model(eval_inputs[0]).cpu().numpy()
                        e_target_numpy = eval_inputs[1].cpu().numpy()
                        e_seq_len_numpy = eval_inputs[2].cpu().numpy()
                        e_w_a_tracking.append(cal_word_acc(e_pred_numpy, e_target_numpy, e_seq_len_numpy))
                        e_s_a_tracking.append(cal_sen_acc(e_pred_numpy, e_target_numpy, e_seq_len_numpy))

                    logging.info('\n\n------------------ \tEvaluation\t------------------')
                    # logging.info('Step: %s', step)
                    logging.info('Number of batchs: %s', len(e_w_a_tracking))
                    my_logger.add_scalar(e_w_a_mean_tag, np.mean(e_w_a_tracking), step)
                    my_logger.add_scalar(e_s_a_mean_tag, np.mean(e_s_a_tracking), step)
                    my_logger.add_scalar(eval_duration_tag, time.time()-start, step)
                    my_logger.add_scalar(e_w_a_std_tag, np.std(e_w_a_tracking), step)
                    my_logger.add_scalar(e_s_a_sdt_tag, np.std(e_s_a_tracking), step)
                    # logging.info('w_a: %.4f±%.4f \t s_a: %.4f±%.4f \t Duration: %.4f s/step',
                    #              e_w_a_tracking.mean(), float(np.std(e_w_a_tracking.figures)),
                    #              e_s_a_tracking.mean(), float(np.std(e_s_a_tracking.figures)),
                    #              time.time() - start)

                    training_checker.update(np.mean(e_w_a_tracking), step)
                    best_score, best_score_step = training_checker.best()
                    logging.info('Current best score: %s recorded at step %s', best_score, best_score_step)

                    eval_inputs = next(iter(eval_loader))
                    eval_inputs = [item.to(device) for item in eval_inputs]
                    predict_and_print_sample(*eval_inputs)


