from UGATIT import UGATIT
import argparse
from utils import *
import wandb
import socket

"""parsing and configuration"""

autodlpath = '/tmp/pycharm_project_837/'


def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=True, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='IRAY', help='dataset_name')

    parser.add_argument('--iteration', type=int, default=100000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=10000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight for Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='/tmp/pycharm_project_837/results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)

    parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
    parser.add_argument('--lambda_NCE', type=int, default=10, help='weight for NCE loss: NCE(G(X), X)')
    parser.add_argument('--nce_idt', type=cut_str2bool, nargs='?', const=True, default=False,
                        help='use NCE loss for identity mapping: NCE(G(Y), Y))')
    parser.add_argument('--nce_layers', type=str, default='0,4,8,12', help='compute NCE loss on which layers')
    parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                        type=cut_str2bool, nargs='?', const=True, default=False,
                        help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
    parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                        help='how to downsample the feature map')
    parser.add_argument('--netF_nc', type=int, default=256)
    parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
    parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
    parser.add_argument('--flip_equivariance',
                        type=cut_str2bool, nargs='?', const=True, default=False,
                        help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

    parser.set_defaults(pool_size=0)  # no image pooling

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')
        # if os.path.exists(autodlpath+'tamp_record.txt'):
        #     # 如果文件已经存在，则以追加模式打开
        #     mode = 'a'
        # else:
        #     # 如果文件不存在，则创建文件并打开
        #     mode = 'w'
        # with open('tamp_record.txt', mode) as f:
        #     f.write('number of epochs must be larger than or equal to one\n')
        #     f.close()

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
        # if os.path.exists(autodlpath+'tamp_record.txt'):
        #     # 如果文件已经存在，则以追加模式打开
        #     mode = 'a'
        # else:
        #     # 如果文件不存在，则创建文件并打开
        #     mode = 'w'
        # with open(autodlpath+'tamp_record.txt', mode) as f:
        #     f.write('batch size must be larger than or equal to one\n')
        #     f.close()
    return args

"""main"""
def main():
    torch.cuda.empty_cache()

    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = UGATIT(args)

    # build graph
    gan.build_model()

    if args.phase == 'train' :
        gan.train()
        print(" [*] Training finished!\n")
        if os.path.exists('tamp_record.txt'):
            # 如果文件已经存在，则以追加模式打开
            mode = 'a'
        else:
            # 如果文件不存在，则创建文件并打开
            mode = 'w'
        with open('tamp_record.txt', mode) as f:
            f.write(" [*] Training finished!\n")
            f.close()

    if args.phase == 'test' :
        gan.test()
        print(" [*] Test finished!")

if __name__ == '__main__':
    main()
