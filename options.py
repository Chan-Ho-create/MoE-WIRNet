import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--epochs', type=int, default=120, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=8,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['HAZE', 'RAIN', 'SNOW'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='/root/autodl-tmp/zhushiyan1_WeatherBench/data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--drs_dir', type=str, default='/root/autodl-tmp/Dataset/train/HAZE/', help='where training images of HAZE saves.')
parser.add_argument('--drd_dir', type=str, default='/root/autodl-tmp/Dataset/train/RAIN/', help='where training images of RAIN saves.')
parser.add_argument('--nrs_dir', type=str, default='/root/autodl-tmp/Dataset/train/SNOW/', help='where training images of SNOW saves.')
parser.add_argument('--output_path', type=str, default="Output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Derain/", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="coba",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="weatherbench_ckpt",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 1,help = "Number of GPUs to use for training")
parser.add_argument("--val_dir", type=str, default="/root/autodl-tmp/Dataset/test/", help="Path to the root validation dataset containing HAZE, RAIN, SNOW.")
parser.add_argument("--val_interval", type=int, default=1, help="Validate every N epochs.")

options = parser.parse_args()

