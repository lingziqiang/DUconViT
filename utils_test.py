import os
import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from utils import DiceLoss, vis_save
from tqdm import tqdm
from tools.metrics import get_all_metrics
from tools.metrics import DSC as get_DSC
from tools.metrics import IOU as get_IOU
from scipy.ndimage.interpolation import zoom
from torchvision.utils import save_image



def test_valid(args, test_model, num_classes=2):
    from datasets.dataset import Osteosarcoma_dataset

    db_test = Osteosarcoma_dataset(base_dir=args.valid_dataset_path, split="test_vol", list_dir=args.list_dir,)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    metric_value_total = np.array([0.0 for i in range(7)])  # acc, prece, recall, f1, DSC_list, HM_list, IOU_list
    save_loss = 0
    image_num = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        #image、label:（num_classes, Batch_size, H, W）、（1, Batch_size, H, W）
        image, label = sampled_batch["image"], sampled_batch["label"]

        for i_image in range(image.shape[1]):
            image_num+=1
            with torch.no_grad():
                one_image = zoom(image[0][i_image].detach().numpy(),
                                 (args.img_size/image[0][i_image].shape[0], args.img_size/image[0][i_image].shape[1]),
                                 order=3)
                one_label = zoom(label[0, i_image].detach().numpy(),
                                 (args.img_size/label[0, i_image].shape[0], args.img_size/label[0, i_image].shape[1]),
                                 order=0)

                one_image, one_label = torch.Tensor(one_image), torch.Tensor(one_label)
                if args.n_gpu >= 1:
                    one_image, one_label = one_image.cuda(), one_label.cuda()
                else:
                    one_image, one_label = one_image.to(torch.device('cpu')), one_label.to(torch.device('cpu'))
                outputs = test_model(one_image.unsqueeze(0).unsqueeze(0))  # （H,W）to（Batch, channels, H, W）
                # outputs' shape:（Batch_Size, n_classes, W, H）
                new_outputs = torch.softmax(outputs, dim=1)
                new_outputs = new_outputs.argmax(dim=1)  # （Batch， W, H），class:0~num_classes-1

                ce_loss_func = CrossEntropyLoss()
                dice_loss_func = DiceLoss(n_classes=2)
                ce_loss = ce_loss_func(outputs, one_label.unsqueeze(0).long())
                dice_loss = dice_loss_func(outputs, one_label.unsqueeze(0), softmax=True)
                loss = 0.4 * ce_loss + 0.6 * dice_loss

                if args.n_gpu > 0:
                    new_outputs = new_outputs.cpu()
                    one_label = one_label.cpu()
                accuraccy, precision, recall, f1, DSC, HM, IOU = get_all_metrics(new_outputs.detach().numpy(),
                                                                                 one_label.detach().numpy())
                # print([accuraccy, precision, recall, f1, DSC, HM, IOU])
                metric_value_total += np.array([accuraccy, precision, recall, f1, DSC, HM, IOU])
                save_loss += loss.item()
    metrics = metric_value_total/image_num
    save_loss /= image_num
    metrics = np.insert(metrics, 0, save_loss)

    return metrics

#对2类别分割图像单独保存分割图像
def save_test_image(args, test_model, save_path, split="all_data",numclasses=2):
    from datasets.dataset import Osteosarcoma_dataset

    db_test = Osteosarcoma_dataset(base_dir=args.valid_dataset_path, split=split, list_dir=args.list_dir, )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    image_num = 0
    dsc_all = 0
    iou_all = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label = sampled_batch["image"], sampled_batch["label"]

        for i_image in range(image.shape[1]):
            image_num += 1
            with torch.no_grad():
                one_image = zoom(image[0][i_image].detach().numpy(),
                                 (args.img_size / image[0][i_image].shape[0],
                                  args.img_size / image[0][i_image].shape[1]),
                                 order=3)
                one_label = zoom(label[0, i_image].detach().numpy(),
                                 (args.img_size / label[0, i_image].shape[0],
                                  args.img_size / label[0, i_image].shape[1]),
                                 order=0)
                one_image, one_label = torch.Tensor(one_image), torch.Tensor(one_label)

                if args.n_gpu >= 1:
                    one_image, one_label = one_image.cuda(), one_label.cuda()
                else:
                    one_image, one_label = one_image.to(torch.device('cpu')), one_label.to(torch.device('cpu'))
                outputs = test_model(one_image.unsqueeze(0).unsqueeze(0))

                new_outputs = torch.softmax(outputs, dim=1)
                new_outputs = new_outputs.argmax(dim=1)

                if args.n_gpu > 0:
                    new_outputs = new_outputs.cpu()
                    one_label = one_label.cpu()
                DSC = get_DSC(new_outputs.detach().numpy(), one_label.detach().numpy())
                IOU = get_IOU(new_outputs.detach().numpy(), one_label.detach().numpy())
                dsc_all+=DSC
                iou_all+=IOU


                _image = (one_image/255).unsqueeze(0)
                _mask = one_label.unsqueeze(0)
                _out_image = new_outputs

                name = str(image_num) + '_dsc_' + str(DSC)[:5]

                img = torch.stack([_image.cpu(), _mask.cpu(), _out_image.cpu()], dim=0)
                save_image(img, os.path.join(save_path, f"{name}.png"))

    mean_dsc = dsc_all / image_num
    mean_iou = iou_all / image_num
    print(f"mean DSC：{mean_dsc:.3f}, mean IOU：{mean_iou:.3f}")


def calculate_scores(args, test_model, save_path, split="all_data",numclasses=2):
    from datasets.dataset import Osteosarcoma_dataset
    db_test = Osteosarcoma_dataset(base_dir=args.valid_dataset_path, split=split, list_dir=args.list_dir, )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    image_num = 0
    dsc_all = 0
    iou_all = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label = sampled_batch["image"], sampled_batch["label"]

        for i_image in range(image.shape[1]):
            image_num += 1
            with torch.no_grad():
                one_image = zoom(image[0][i_image].detach().numpy(),
                                 (args.img_size / image[0][i_image].shape[0],
                                  args.img_size / image[0][i_image].shape[1]),
                                 order=3)
                one_label = zoom(label[0, i_image].detach().numpy(),
                                 (args.img_size / label[0, i_image].shape[0],
                                  args.img_size / label[0, i_image].shape[1]),
                                 order=0)
                one_image, one_label = torch.Tensor(one_image), torch.Tensor(one_label)
                if args.n_gpu >= 1:
                    one_image, one_label = one_image.cuda(), one_label.cuda()
                else:
                    one_image, one_label = one_image.to(torch.device('cpu')), one_label.to(torch.device('cpu'))
                outputs = test_model(one_image.unsqueeze(0).unsqueeze(0))

                new_outputs = torch.softmax(outputs, dim=1)
                new_outputs = new_outputs.argmax(dim=1)

                if args.n_gpu > 0:
                    new_outputs = new_outputs.cpu()
                    one_label = one_label.cpu()
                DSC = get_DSC(new_outputs.detach().numpy(), one_label.detach().numpy())
                IOU = get_IOU(new_outputs.detach().numpy(), one_label.detach().numpy())
                dsc_all+=DSC
                iou_all+=IOU


                _image = (one_image/255).unsqueeze(0)
                _mask = one_label.unsqueeze(0)
                _out_image = new_outputs

                name = str(image_num) + '_dsc_' + str(DSC)[:5]

                img = torch.stack([_image.cpu(), _mask.cpu(), _out_image.cpu()], dim=0)
                save_image(img, os.path.join(save_path, f"{name}.png"))

    mean_dsc = dsc_all / image_num
    mean_iou = iou_all / image_num
    print(f"mean DSC：{mean_dsc:.3f}, mean IOU：{mean_iou:.3f}")