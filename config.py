
import argparse

def get_args():
  """
      The argument parser
  """
  parser = argparse.ArgumentParser()
  #2019
  #72309
  parser.add_argument('--random_seed', type=int, default=2019, help='Random seed')

  parser.add_argument('--dataset_dir', type=str, default='../../sim6.0', help='dataset name')
  parser.add_argument('--log_dir', type=str, default='save_sim6.0_bat/save_sim6.0_bat_20201016/logs', help='log path')
  parser.add_argument('--save_dir', type=str, default='save_sim6.0_bat/save_sim6.0_bat_20201016/saves', help='save path')
  parser.add_argument('--img_dir', type=str, default='save_sim6.0_bat/save_sim6.0_bat_20201016/imgs', help='save path')
  parser.add_argument('--eval_save', type=str, default='save_sim6.0_bat/save_sim6.0_bat_20201016/eval', help='eval images save directory')
  parser.add_argument('--test_save', type=str, default='result/test', help='test images save directory')

  parser.add_argument('--input_size', type=list, default=[180, 240], help='dims of input data')
  parser.add_argument('--event_channel', type=int, default=8, help="channels of event map")
  parser.add_argument('--gray_channel', type=int, default=1, help="channels of gray")
  parser.add_argument('--L1_lambda', type=float, default=10.0, help="weight on identity term in objective")
  parser.add_argument('--ngen_filters', type=int, default=32, help="number of generator filters")
  parser.add_argument('--ndis_filters', type=int, default=32, help="number of discriminator filters")

  parser.add_argument('--pool_size', type=int, default=50, help='image pool size')
  parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
  parser.add_argument('--decay_start', type=int, default=100, help='anneal start epoch')
  parser.add_argument('--nb_epoch', type=int, default=200, help='The number of epoch')
  parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
  parser.add_argument('--max_grad_norm', type=float, default=10.0, help='Max norm of gradient')
  parser.add_argument('--use_lsgan', type=bool, default=True, help='whether to use lsgan')
  parser.add_argument('--use_srgan', type=bool, default=False, help='whether to use srgan discriminator')
  parser.add_argument('--print_step', type=int, default=100, help='number step to print')
  parser.add_argument('--save_step', type=int, default=10000, help='number epoch to save')

  return parser.parse_args()
