import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import random
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import OrderedDict
from time import time, ctime, localtime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset,DataLoader
import cv2
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
import timm
import math
import pydicom as dcm
from sklearn.model_selection import train_test_split,train_test_split
from sklearn.model_selection import StratifiedKFold,  KFold
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
set_seed(0)
train_dir = 'dicom_2_LAT/'
save_dir = 'roc_curve'
classes = []
#무슨 클래스가 있는지 출력
for i in range(len(glob.glob(train_dir+'/*'))):
    classes.append(glob.glob(train_dir+'/*')[i].split('/')[-1])
class_num = len(classes)
print('Total Class num: ',class_num)
print('Class label:')
classes=sorted(classes)

for i in range(class_num):print('{:5d}th : ' .format(i+1),classes[i])



# 데이터 전처리
train_transforms = transforms.Compose([
                                       transforms.Resize((600,380)),
                                       transforms.RandomHorizontalFlip(p=0.4),
                                       transforms.RandomRotation(degrees=30),
                                       # transforms.RandomAffine((-15,15), translate=(0.1, 0.1)),
                                       # transforms.RandomVerticalFlip(p=0.2),
                                       transforms.RandAugment(2,9),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])
val_transforms = transforms.Compose([
                                       transforms.Resize((600,380)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(train_dir, transform=val_transforms)

#인덱스를 이용해 train_valid 나눔
valid_size = 0.2
batch_size = 12
num_train = len(train_data)
num_train = int(num_train)
indices = list(range(num_train))
np.random.shuffle(indices) # 순서를 랜덤으로 섞음
split = int(np.floor(valid_size * num_train)) # 데이터의 0.2에 할당하는 값
train_idx, valid_idx = indices[split:], indices[:split] # split을 기준으로 나눔


#train_sampler = SubsetRandomSampler(train_idx) # 0.8 할당
#valid_sampler = SubsetRandomSampler(valid_idx) # 0.2 할당

#train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    #sampler=train_sampler)
#valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
    #sampler=valid_sampler)


def roc_curve_plot(y_test , pred_proba_c1 , fold):
    # 임곗값에 따른 FPR, TPR 값을 반환 받음. 
    fprs , tprs , thresholds = roc_curve(y_test ,pred_proba_c1)

    # ROC Curve를 plot 곡선으로 그림. 
    plt.plot(fprs , tprs, label='ROC')
    # 가운데 대각선 직선을 그림. 
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
  
    # FPR X 축의 Scale을 0.1 단위로 변경, X,Y 축명 설정등   
    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"best_rocLAT600_{fold}.png"))
    plt.close()
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    # pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=0.2, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device) # 배치 내에 랜덤한 사진
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# 학습 함수
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []
    bar = tqdm(train_loader)
    for i, (inputs, labels) in enumerate(bar):
        r = np.random.random()
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # if r > 0.5:  # mixup
        #     x, targets_a, targets_b, lam = mixup_data(inputs, labels, 0.2, True)
        #     outputs = model(x)
        #     loss = mixup_criterion(criterion, outputs.squeeze(), targets_a.float(), targets_b.float(), lam)
        # else:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*inputs.size(0)
        y_true += labels.cpu().numpy().tolist()
        y_pred += (outputs.sigmoid().cpu().detach().numpy() > 0.5).astype(int).tolist()
        y_prob += (outputs.sigmoid().cpu().detach().numpy()).tolist()
        loss_np = loss.detach().cpu().numpy()
        bar.set_description('loss: %.5f' % (loss_np))

    train_loss = running_loss / len(train_loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    return train_loss, acc, auc, f1

# 검증 함수
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(val_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())

            running_loss += loss.item()*inputs.size(0)
            y_true += labels.cpu().numpy().tolist()
            y_pred += (outputs.sigmoid().cpu().detach().numpy() > 0.5).astype(int).tolist()
            y_prob += (outputs.sigmoid().cpu().detach().numpy()).tolist()
    val_loss = running_loss / len(val_loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    return val_loss, acc, auc, f1, y_true, y_prob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = models.efficientnet_v2_m(pretrained=True)
#model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                       #torch.nn.Linear(in_features=1280, out_features=1,bias= True))
#model = nn.DataParallel(model)
#model = model.to(device)

#criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=0.00003)

model_path = 'weights/eff_model.pt'

# 학습
num_epochs = 20
#scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, num_epochs - 1)
#scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
# best_val_loss = float('inf')
# best_val_acc = float('inf')*-1
kf = StratifiedKFold(n_splits=5, shuffle=True)  
#sig = nn.Sigmoid()
#best_auc_list = []
for fold, (train_ind, valid_ind) in enumerate(kf.split(train_data,train_data.targets)): 
    print('Starting fold = ', fold+1)
    best_val_loss = float('inf')
    best_val_acc = float('inf') * -1
    
    train_sampler_kfold = SubsetRandomSampler(train_ind) 
    valid_sampler_kfold = SubsetRandomSampler(valid_ind) 
    train_loader_kfold = torch.utils.data.DataLoader(train_data, batch_size=batch_size, #trainloader
    sampler=train_sampler_kfold,drop_last=True)
    valid_loader_kfold = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, #trainloader
    sampler=valid_sampler_kfold,drop_last=True)
    model = models.efficientnet_v2_s(pretrained=True)
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
    # model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1).to(device)
    # ckpt = torch.load('C:/Users/MMC/Downloads/model_AP0.pt')
    # model.load_state_dict(ckpt, strict=False)
    #model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 12], gamma=0.3)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, num_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=scheduler_cosine)
    for epoch in range(num_epochs):
        train_loss, train_acc, train_auc, train_f1 = train(model, train_loader_kfold, optimizer, criterion, device)
        val_loss, val_acc, val_auc, val_f1, y_true, y_prob = validate(model, valid_loader_kfold, criterion, device)
        # scheduler_warmup.step()
        scheduler_warmup.step()
        if epoch == 1:
            scheduler_warmup.step()
        print(f'Epoch {epoch+1}/{num_epochs},lr: {optimizer.param_groups[0]["lr"]:.7f} , Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
    
    
        #if val_loss < best_val_loss:
            #torch.save(model.state_dict(), model_path)
            #best_val_loss = val_loss
        if best_val_acc <= val_acc: # epoch가 더 낮지만 최소였던 지점을 저장하기 위함
            print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            best_val_acc,
            val_acc))
            torch.save(model.state_dict(), f'model_finaldata_effLAT600{fold}.pt')
            best_val_acc = val_acc
            roc_curve_plot(y_true, y_prob, fold+1) # 더 최적의 값을 저장
        #if val_loss <= best_val_loss: # epoch가 더 낮지만 최소였던 지점을 저장하기 위함
           # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            #best_val_loss,
            #val_loss))
            #torch.save(model.state_dict(), f'model_effv2{fold}.pt') # 이 과정에선 크게 의미가 없음.
            #best_val_loss = val_loss # 더 최적의 값을 저장

print('Training finished')