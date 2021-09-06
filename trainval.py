import argparse
import ast
import os

import torch
import datetime
import logging
import random
import yaml
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from src.processor import processor

# Use Deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser(description='STAR')
    parser.add_argument('--dataset', default='trajectory_combined')
    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--config')
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    parser.add_argument('--test_set', default='trajectory_combined', type=str,
                        help='Set this value to [trajectory_combined, eth, hotel, zara1, zara2, univ] for ETH-univ, ETH-hotel, UCY-zara01, UCY-zara02, UCY-univ')
    parser.add_argument('--base_dir', default='.', help='Base directory including these scripts.')
    parser.add_argument('--save_base_dir', default='./output/', help='Directory for saving caches and models.')
    parser.add_argument('--phase', default='train', help='Set this value to \'train\' or \'test\'')
    parser.add_argument('--train_model', default='star', help='Your model name')
    parser.add_argument('--load_model', default=None, type=str, help="load pretrained model for test or training")
    parser.add_argument('--model', default='star.STAR')
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=8, type=int)
    parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--batch_around_ped', default=256, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--show_step', default=100, type=int)
    parser.add_argument('--start_test', default=10, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval)
    parser.add_argument('--ifsave_results', default=False, type=ast.literal_eval)
    parser.add_argument('--randomRotate', default=True, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--neighbor_thred', default=10, type=int)
    parser.add_argument('--learning_rate', default=0.0015, type=float)
    parser.add_argument('--clip', default=1, type=int)
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)

    return parser


def load_arg(p):
    # save arg
    if os.path.exists(p.config):
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    curr_time = datetime.datetime.now()
    output_dir = f"exp_{p.test_set}_{curr_time.strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, f"IndividualTF_ckpts")
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_output_dir = os.path.join(output_dir, f"IndividualTF_outputs")
    os.makedirs(model_output_dir, exist_ok=True)

    # keep track of console outputs and experiment settings
    set_logger(os.path.join(output_dir, f"train_{p.test_set}.log"))
    config_file = open(
        os.path.join(output_dir, f"config_{p.test_set}.yaml"), "w"
    )
    yaml.dump(p, config_file)
    tensorboard_dir = os.path.join(output_dir, "tensorboard")
    logger = SummaryWriter(tensorboard_dir)

    seed = 72
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    p.save_dir = output_dir
    p.model_dir = checkpoint_dir
    # p.config = p.model_dir + '/config_' + p.phase + '.yaml'

    # if not load_arg(p):
        # save_arg(p)

    # args = load_arg(p)

    torch.cuda.set_device(0)

    trainer = processor(p, logger)

    if p.phase == 'test':
        trainer.test()
    else:
        trainer.train()
