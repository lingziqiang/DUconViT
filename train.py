import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import SwinUnet as ViT_seg
from networks.Unet_SwinUnet import Unet_SwinUnet
from networks.Unet_using_SwinUnet_as_Bottleneck import Unet_using_SwinUnet_as_Bottleneck
from networks.Unet_connect_SwinUnet_in_the_ending import Unet_connect_SwinUnet_in_the_ending
from networks.Swin_Unet_Then_Unet import Swin_Unet_Then_Unet
from networks.transunet import VisionTransformer as Transunet
from networks.utnet import UTNet
from networks.attention_Unet import AttUNet
from networks.UNetPlusPlus import NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from networks.MedT import MedT
# from networks.MCTrans import MCTrans_all as MCTrans
from networks.FCN import FCN8s
from config import get_config
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from utils import DiceLoss, vis_save
from tqdm import tqdm
from utils_test import test_valid, save_test_image
from tools.metrics import get_all_metrics
from tools.excel_operation import to_Excel

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../0.data/npz_h5_data/train_npz', help='root dir for data')
parser.add_argument('--valid_dataset_path', type=str,
                    default='../0.data/npz_h5_data/test_vol_h5', help='root dir for validation  data')
parser.add_argument('--dataset', type=str,
                    default='Osteosarcoma', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Osteosarcoma', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str,  default='./results', help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iterations number per epoch to train')
parser.add_argument('--max_epochs', type=int,
                    default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--max_iou', type=float, default=0.00,
                    help='max iou')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--model', type=str, default='Swin_Unet', help='selected model for training')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument('--checkpoint_path', type=str, default=None, help='Get trained parameters')
parser.add_argument('--start_epoch', type=int, default=1, help='epoch at start')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)


def trainer(args, model):
    from datasets.dataset import Osteosarcoma_dataset, RandomGenerator


    base_lr = args.base_lr
    num_classes = args.num_classes
    if args.n_gpu >= 1:
        batch_size = args.batch_size * args.n_gpu
    else:
        batch_size = args.batch_size

    data_base = Osteosarcoma_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                     transform=transforms.Compose(
                                         [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(data_base, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)  # 将num_workers改为0

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss_func = CrossEntropyLoss()
    dice_loss_func = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    start_epoch = args.start_epoch
    max_epoch = args.max_epochs - start_epoch
    trainloader_len = len(trainloader)
    max_iterations = args.max_epochs * len(trainloader)
    print("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)

    max_iou = args.max_iou
    for epoch_num in iterator:
        image_num = 0
        save_loss = 0
        batch_num = len(trainloader)
        metric_value_total = np.array([0.0 for i in range(7)])  # acc, prece, recall, f1, DSC_list, HM_list, IOU_list
        for i_batch, data_batch in enumerate(trainloader):
            image_batch, label_batch = data_batch['image'], data_batch['label']

            if args.dataset == 'Osteosarcoma':
                label_batch = torch.where(label_batch > 0, torch.ones_like(label_batch), label_batch)

            if args.n_gpu >= 1:
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            else:
                image_batch, label_batch = image_batch.to(torch.device('cpu')), label_batch.to(torch.device('cpu'))

            outputs = model(image_batch)  #（Batch_Size, n_classes, W, H）
            ce_loss = ce_loss_func(outputs, label_batch[:].long())
            dice_loss = dice_loss_func(outputs, label_batch, softmax=True)
            loss = 0.4 * ce_loss + 0.6 * dice_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''
            outputs' size:（Batch_Size, n_classes, W, H）
            softmax（output, dim=1）then argmax(output, dim=1)
            get 0~n_classes-1
            '''
            new_outputs = torch.softmax(outputs, dim=1)
            new_outputs = new_outputs.argmax(dim=1)  # （Batch， W, H）
            if args.n_gpu > 0:
                image_batch = image_batch.cpu()
                new_outputs = new_outputs.cpu()
                label_batch = label_batch.cpu()

            for i_image in range(outputs.shape[0]):
                vis_save(image_batch[i_image][0].detach().numpy(), new_outputs[i_image].detach().numpy(),
                         os.path.join(args.output_dir, 'predict'), f'{image_num}.jpg')
                image_num += 1

                # only useful for num_classes=2
                accuraccy, precision, recall, f1, DSC, HM, IOU = get_all_metrics(new_outputs[i_image].detach().numpy(),
                                                                                 label_batch[i_image].detach().numpy())
                metric_value_total += np.array([accuraccy, precision, recall, f1, DSC, HM, IOU])
                # print([accuraccy, precision, recall, f1, DSC, HM, IOU])
            save_loss += loss.item()


            lr_ = base_lr * (1.0 - (iter_num + start_epoch * trainloader_len) / max_iterations) ** 0.9
            if args.start_epoch + epoch_num < 50:
                lr_ = 0.05
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            print('iteration %d : loss : %f, loss_ce: %f, dice_loss: %f' % (
            iter_num, loss.item(), ce_loss.item(), dice_loss.item()))


        metrics = metric_value_total / image_num
        metrics = np.insert(metrics, 0, save_loss / batch_num)
        val_metrics = test_valid(args, model)
        all_metrics = np.append(metrics, val_metrics)
        print("all_metrcis:", all_metrics)
        to_Excel('./results/metrics.xlsx', all_metrics, args.start_epoch + epoch_num)

        save_interval = 30
        true_epoch_num = args.start_epoch + epoch_num
        if true_epoch_num % save_interval == 0 or true_epoch_num >= args.max_epochs - 1:
            # save_mode_path = os.path.join(args.output_dir, 'epoch_' + str(true_epoch_num) + '.pth')
            # torch.save(model.state_dict(), save_mode_path)
            # print("第" + str(iter_num) + "次迭代，已保存：" + 'epoch_' + str(true_epoch_num) + '.pth')
            save_mode_path = os.path.join(args.output_dir, args.model + '_the_last_epoch' + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            print("第" + str(iter_num) + "次迭代，已保存：" + args.model + '_the_last_epoch' + '.pth')
        test_IOU = val_metrics[-1]
        if test_IOU > max_iou:
            max_iou = test_IOU
            save_mode_path = os.path.join(args.output_dir, args.model+'_the_best_epoch' + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            print("The " + str(iter_num) + " epoch，saved：" + args.model+'_the_best_epoch' + '.pth')


    #save the weights of the last epoch
    save_mode_path = os.path.join(args.output_dir, args.model + '_the_last_epoch' + '.pth')
    torch.save(model.state_dict(), save_mode_path)
    print("The" + str(iter_num) + " epoch，saved：" + args.model+'_the_last_epoch' + '.pth')

dataset_config = {
    'Osteosarcoma': {
        'root_path': args.root_path,
        'list_dir': args.list_dir,
        'num_classes': 2
    }
}
dataset_name = args.dataset
args.num_classes = dataset_config[dataset_name]['num_classes']
args.root_path = dataset_config[dataset_name]['root_path']
args.list_dir = dataset_config[dataset_name]['list_dir']

if __name__ == '__main__':
    device = torch.device('cuda' if (torch.cuda.is_available() and args.n_gpu > 0) else 'cpu')
    if args.model == 'Swin_Unet':
        net = ViT_seg(config, num_classes=args.num_classes).to(device)
    elif args.model == 'Unet_SwinUnet':
        net = Unet_SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes, in_channels=1, dock_channels=128).to(device)
    elif args.model == 'Unet_using_SwinUnet_as_Bottleneck':
        net = Unet_using_SwinUnet_as_Bottleneck(config, img_size=args.img_size, out_channel=args.num_classes, dock_channel=256).to(device)
    elif args.model == 'Unet_connect_SwinUnet_in_the_ending':
        net = Unet_connect_SwinUnet_in_the_ending(config, img_size=args.img_size, out_channel=args.num_classes, in_channel=1).to(device)
    elif args.model == 'Swin_Unet_Then_Unet':
        net  = Swin_Unet_Then_Unet(config, img_size=args.img_size, num_classes=args.num_classes, in_channels=3, dock_channels=256).to(device)
    elif args.model =='Transunet':
        from networks.transunet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        net = Transunet(config_vit, img_size=224, num_classes=4).to(device)
    elif args.model == 'UTNet':
        net = UTNet(1, 32, 2, reduce_size=2,
                    block_list='1234', num_blocks=[1, 1, 1, 1], num_heads=[4, 4, 4, 4],
                    projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False,
                    maxpool=True).to(device)
    elif args.model == 'AttUNet':
        net =AttUNet(1, 2).to(device)
    elif args.model == 'UNet++':
        net = NestedUNet(1, 2).to(device)
    elif args.model == 'MedT':
        net = MedT(img_size=args.img_size, imgchan=1, num_classes=args.num_classes).to(device)
    # elif args.model == 'MCTrans':
    #     from networks.mctrans.mctrans_config import model as MCTrans_cfg
    #     net = MCTrans(MCTrans_cfg, num_classes=args.num_classes).to(device)
    elif args.model == 'FCN':
        net = FCN8s(input_channel=1, n_class=args.num_classes).to(device)
    elif args.model == 'UNet':
        net = U_Net(1, 2).to(device)


    if args.checkpoint_path == None:  # 如果没有断点，就载入预训练模型 只有224x224的预训练模型的
        # net.load_from(config)
        pass
    else:  # 如果存在断点，进行断点续训KeyError: 'EncoderDecoder is not in the network registry'
        net.load_state_dict(torch.load(args.checkpoint_path))

    '''
    train
    '''
    trainer(args, net)
    '''
    test and save predict image code
    '''
    # only_predict_save_path_all =  './results/all_predict'
    # only_predict_save_path_test = './results/test_predict'
    # save_test_image(args, net, only_predict_save_path_all, split="all_data")
    # save_test_image(args, net, only_predict_save_path_test, split="test_vol")
    # scores = test_valid(args, net)
    # print(f"loss:{scores[0]}, acc:{scores[1]}, pre:{scores[2]}：,recall:{scores[3]},f1-score:{scores[4]}, "
    #       f"dsc{scores[5]}, hm:{scores[6]}, iou:{scores[7]}")
