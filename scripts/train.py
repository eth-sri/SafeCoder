import os
import argparse

from safecoder.utils import set_seed, set_logging
from safecoder.trainer import Trainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)

    parser.add_argument('--pretrain_name', type=str, default='codegen-350m')

    # sec and sec-instruct
    parser.add_argument('--loss_weight', type=float, default=1.0)

    # sven prefix-tuning
    parser.add_argument('--sven', action='store_true', default=False)

    # training arguments 
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_acc_steps', type=int, default=16)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--kl_loss_weight', type=int, default=0) # will be divided by 1000
    parser.add_argument('--exclude_neg', action='store_true', default=False)
    parser.add_argument('--no_weights', action='store_true', default=False)

    # lora arguments
    parser.add_argument('--lora', action='store_true', default=False, help='Toggle to use lora in training')
    parser.add_argument('--r', type=int, default=16, help='Lora hidden dimensions')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Alpha param, see Lora doc.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout in the learned extensions')
 
    # upsampling arguments
    """
    --sampling_size:
        the size of sampling, <=0 means no sampling
        dataset.Upsampler._upsample_all_prob: the percentange of the sampled sec dataset compared to the func dataset
        dataset.Upsampler._upsample_minority: sample classes with <sampling_size samples to sampling_size
    --sampling_weight:
        select the mode of how the sampling weight of each cwe at lang is calcualted when doing -all sempling modes:
            uniform: each example is treated equally and uniform sampling is employed across the whole dataset
            inverse-prop: cwes with less example are more likely to get sampled, chosing this balances the cwes
    --cwes:
        select a list of the cwes you want to upsample, or select all
    --langs:
        select a list of the langs you want to include in upsampling, or select all
    """
    parser.add_argument('--sampling_size', type=int, default=-1)
    parser.add_argument('--sampling_method', type=str, choices=['uniform', 'inverse-prop', 'minority'], default='minority')
    parser.add_argument('--cwes', type=str, nargs='*', default=['all'])
    parser.add_argument('--langs', type=str, nargs='*', default=['all'])

    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2)

    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    parser.add_argument('--model_dir', type=str, default='../trained/')

    args = parser.parse_args()

    # adjust the naming to make sure that it is in the expected format for loading
    if args.lora and not args.output_name.startswith(f'{args.pretrain_name}-lora'):
        args.output_name = f'{args.pretrain_name}-lora-' + args.output_name

    if args.sven and not args.output_name.startswith(f'{args.pretrain_name}-sven'):
        args.output_name = f'{args.pretrain_name}-sven-' + args.output_name

    if args.sampling_size == -1 and 'lmsys' in args.datasets:
        args.sampling_size = 40

    if args.sampling_size == -1 and 'evol' in args.datasets:
        args.sampling_size = 20

    if args.num_train_epochs is None:
        if args.sven:
            args.num_train_epochs = 5
        else:
            if args.pretrain_name.startswith('codellama'):
                args.num_train_epochs = 5
            else:
                args.num_train_epochs = 2

    if args.learning_rate is None:
        if args.sven:
            if args.pretrain_name.startswith('starcoderbase'):
                args.learning_rate = 5e-2
            else:
                args.learning_rate = 1e-2
        else:
            if args.pretrain_name.startswith('codellama'):
                args.learning_rate = 1e-3
            else:
                args.learning_rate = 2e-5

    if args.exclude_neg:
        args.sampling_size = args.sampling_size // 2

    args.output_dir = os.path.join(args.model_dir, args.output_name)

    return args

def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_seed(args.seed)
    Trainer(args).run()

if __name__ == '__main__':
    main()