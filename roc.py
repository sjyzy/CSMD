import logging
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.optim import SGD
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
import torchvision
import random
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix,classification_report,roc_curve, auc,roc_auc_score
from monai.losses import FocalLoss,DiceLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms.v2 import RandomEqualize,ColorJitter
import wandb
import monai
from monai.data import  ImageDataset, DataLoader 
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity,ScaleIntensityFixedMean,CropForeground,RandRotate90,RandZoom,RandFlip,CenterSpatialCrop,SpatialCrop,RandSmoothFieldAdjustContrast,RandSmoothFieldAdjustIntensity
import torch.nn.functional as F
# python roc.py --name t2sag --img_path /nas_data/SJY/dataset/301hospital/8_1train_val_test/final/image/t2_sag --pth_path /nas_data/SJY/dataset/301hospital/8_1train_val_test/final/pth/id4_t2sag/best_auc.pth --label_path /nas_data/SJY/dataset/301hospital/8_1train_val_test/final/text/high_pain_text.xlsx
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='VADepthNet PyTorch implementation.', fromfile_prefix_chars='@')

    parser.add_argument('--name',                     type=str,   help='name of png', default='')
    parser.add_argument('--img_path',               type=str,   help='path of image', default='')
    parser.add_argument('--pth_path',                 type=str,   help='path of pth', default='')
    parser.add_argument('--label_path',               type=str,   help='path of excel', default='./high_pain_text.xlsx')
    args = parser.parse_args()  # 4、解析参数对象获得解析对象
    return args

args = parse_args()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def threshold_at_one(x): return x >= 10
config = {
    # 数据

    # 'name' : 'id4_t2sag.png',
    # 'title' : 't2 sag',
    # 'image_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/image/t2_sag',
    # 'pth_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/pth/id4_t2sag/best_auc.pth',

    # 'name' : 'id4_t1sag.png',
    # 'title' : 't1 sag',
    # 'image_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/image/t1_sag',
    # 'pth_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/pth/id4_t1sag/best_auc.pth',

    # 'name' : 'id4_t2tra.png',
    # 'title' : 't2 tra',
    # 'image_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/image/t2_tra',
    # 'pth_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/pth/id4_t2tra/best_auc.pth',


    # 'name' : 'id4_ctbone.png',
    # 'title' : 'ct bone',
    # 'image_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/image/ct_bone',
    # 'pth_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/pth/id4_ct_bone/best_auc.pth',

    # 'name' : 'id4_cttissue3.png',
    # 'title' : 'ct tissue',
    # 'image_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/image/ct_tissue',
    # 'pth_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/pth/id4_ct_tissue3/best_accuracy.pth',

    # 'excel_path' : '/nas_data/SJY/dataset/301hospital/8_1train_val_test/final/text/high_pain_text.xlsx',

    # 超参数
    'k':10,
    'epoch_num' : 300,
    'val_interval' : 1,
    'lr' : 1e-4,
    'weight_decay': 1e-8,
    'loss' : torch.nn.CrossEntropyLoss(),
    'model' : monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device),
    'train_transforms' : Compose([
                                CropForeground(select_fn=threshold_at_one),
                                EnsureChannelFirst(), 
                                Resize((320, 320, 32)), 
                                RandRotate90(),
                                RandZoom(),
                                RandFlip(),
                                ]),
    'val_transforms' : Compose([
                                CropForeground(select_fn=threshold_at_one),
                                EnsureChannelFirst(), 
                                Resize((320, 320, 32)),
                                ]),
    
    'batch_size' : 1,
    'num_workers' : 8,
} 

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(3407)


def label_mapping2(label):
    if label == 0:
        return torch.tensor(0)
    else:
        return torch.tensor(1)


def read_data(img_folder,label_file):
    train_label_list = []
    train_img_list = []
    train_id_list = []
    test_label_list = []
    test_img_list = []
    test_id_list = []

    num = 0
    num0 = 0   
    num1 = 0   
    data = pd.read_excel(label_file)
    data.iloc[:, 0] = data.iloc[:, 0].astype(str)
    data.set_index('ID', inplace=True)
    train_id = load_list_from_file('./train_id4.txt')
    test_id = load_list_from_file('./test_id4.txt')

    for root, dirs, files in os.walk(img_folder):
        for file in files:
            if file.endswith(".nii.gz"):
                num += 1
                second_last_directory = os.path.basename(os.path.dirname(os.path.join(root, file)))
                id = second_last_directory.split('_')[-1]
                
                if int(id) in train_id:
                    num0 += 1
                    row_data = data.loc[id].values  
                    train_label_list.append(label_mapping2(row_data[-5]))
                    train_img_list.append(os.path.join(root, file))
                    train_id_list.append(id)
                if int(id) in test_id:
                    num1 += 1
                    row_data = data.loc[id].values  
                    test_label_list.append(label_mapping2(row_data[-5]))
                    test_img_list.append(os.path.join(root, file))
                    test_id_list.append(id)
                   
    print(f'总共{num}个样本,train有{num0}个样本,test有{num1}个样本')
    return train_img_list,train_label_list,train_id_list,test_img_list,test_label_list,test_id_list
               


def bceaccuracy(y_pred,val_labels):
    val_labels = int(val_labels)
    if y_pred>=0:
        y_pred=1
    else:
        y_pred=0
    if y_pred==val_labels:
        return 1
    else:
        return 0

# 将列表保存到文件
def save_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')

# 从文件加载列表数据
def load_list_from_file(filename):
    loaded_list = []
    with open(filename, 'r') as file:
        for line in file:
            loaded_list.append(int(line.strip()))  # 假设列表中的元素是整数

    return loaded_list

def main():
    
    
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
   
    # 301dataset
    train_img_list,train_label_list,train_id_list,test_img_list,test_label_list,test_id_list = read_data(args.img_path,args.label_path)
    best_accuracy = -1
    best_spesificity = -1
    best_auc = -1
    best_accuracy_epoch=0
    best_spesificity_epoch=0
    best_auc_epoch=0

    val_transforms = config["val_transforms"]
    val_ds = ImageDataset(image_files=test_img_list[:], labels=test_label_list[:], transform=val_transforms, image_only = True)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, shuffle=False,pin_memory=torch.cuda.is_available())
    print(device)
   
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(args.pth_path))

    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.15)
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
  
   
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        epoch_val_loss = 0
        con_matrix = 0
        val_outputs_list = []
        val_labels_list = []
        val_outputs_yscore = []
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            val_outputs = model(val_images)
            val_loss = loss_function(val_outputs, val_labels)
            epoch_val_loss += val_loss.item()
            val_outputs_list.append(val_outputs.argmax(dim=1).item())
            val_labels_list.append(val_labels.item())
            y_score = F.softmax(val_outputs, dim=1)[0][1].item()
            val_outputs_yscore.append(y_score)
        print("模型预测",val_outputs_list)
        print("真实标签",val_labels_list)
        auc = roc_auc_score(val_labels_list,val_outputs_list)
        tn, fp, fn, tp = confusion_matrix(val_labels_list,val_outputs_list).ravel()
        fpr, tpr, thresholds = roc_curve(val_labels_list,val_outputs_yscore)
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision_ppv = tp/(tp+fp)
        npv = tn/(tn+fn)
        racell_sensitivity = tp/(tp+fn)
        spesificity = tn/(tn+fp)
        f1 = 2*tp/(2*tp+fp+fn)
        # print(f'fpr:{fpr},tpr:{tpr},thresholds:{thresholds}')
        # print(f'tn:{tn},fp:{fp},fn:{fn},tp:{tp}')
        print("accuracy:{:.4f}, precision_ppv:{:.4f}, npv:{:.4f},racell_sensitivity:{:.4f}, spesificity:{:.4f}, f1:{:.4f}, auc:{:.4f}".format(accuracy, precision_ppv,npv,racell_sensitivity,spesificity,f1,auc))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.savefig(args.name)



    writer.close()
    exit()


if __name__ == "__main__":
    main()
