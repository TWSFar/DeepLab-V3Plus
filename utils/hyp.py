import argparse
import torch

try:
    from torch_utils import select_device
except:
    from .torch_utils import select_device


parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
parser.add_argument('--backbone', type=str, default='resnet',
                    choices=['resnet', 'xception', 'drn', 'mobilenet'],
                    help='backbone name (default: resnet)')
parser.add_argument('--out-stride', type=int, default=16,
                    help='network output stride (default: 8)')
parser.add_argument('--dataset', type=str, default='voc',
                    choices=['voc', 'coco'],
                    help='dataset name (default: pascal)')
parser.add_argument('--use-sbd', action='store_true', default=False,
                    help='whether to use SBD dataset (default: True)')
parser.add_argument('--workers', type=int, default=4,
                    metavar='N', help='dataloader threads')
parser.add_argument('--base-size', type=int, default=513,
                    help='base image size')
parser.add_argument('--crop-size', type=int, default=513,
                    help='crop image size')
parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
parser.add_argument('--loss-type', type=str, default='ce',
                    choices=['ce', 'focal'],
                    help='loss func type (default: ce)')
# training hyper params
parser.add_argument('--epochs', type=int, default=None, metavar='N',
                    help='number of epochs to train (default: auto)')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
parser.add_argument('--batch-size', type=int, default=None,
                    metavar='N', help='input batch size for \
                        training (default: auto)')
parser.add_argument('--test-batch-size', type=int, default=None,
                    metavar='N', help='input batch size for \
                        testing (default: auto)')
parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                    help='whether to use balanced weights (default: False)')
# optimizer params
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (default: auto)')
parser.add_argument('--lr-scheduler', type=str, default='poly',
                    choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: poly)')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    metavar='M', help='w-decay (default: 5e-4)')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='whether use nesterov (default: False)')
# cuda, seed and logging
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# checking point
parser.add_argument('--resume', type=str, default=None,
                    help='resuming path')
parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')
# finetuning pre-trained models
parser.add_argument('--ft', action='store_true', default=False,
                    help='finetuning on a different dataset')
# evaluation option
parser.add_argument('--eval-interval', type=int, default=1,
                    help='evaluuation interval (default: 1)')
parser.add_argument('--no-val', action='store_true', default=False,
                    help='skip validation during training')

args = parser.parse_args()

args.cuda = torch.cuda.is_available()
args.device, args.gpu_ids = select_device(is_head=True)
args.ng = len(args.gpu_ids)

if args.sync_bn is None:
    args.sync_bn = True if args.ng > 1 else False

if args.epochs is None:
    epochs = {
        'coco': 30,
        'voc': 50}
    args.epochs = epochs[args.dataset.lower()]

if args.batch_size is None:
    args.batch_size = min(4 * args.ng, 2)

if args.test_batch_size is None:
    args.test_batch_size = args.batch_size

if args.lr is None:
    lrs = {
        'coco': 0.1,
        'voc': 0.007,
    }
    args.lr = lrs[args.dataset.lower()] / (4 * args.ng) * args.batch_size

if args.checkname is None:
    args.checkname = 'deeplab-' + str(args.backbone)

print('# parametes list:')
for (key, value) in args.__dict__.items():
    print(key, '=', value)
print('')

torch.manual_seed(args.seed)
