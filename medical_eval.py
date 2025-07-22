import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import random
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset,DataLoader, Subset
import cv2
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

save_dir = 'roc_curve_CM'
classes = []
#무슨 클래스가 있는지 출력
# for i in range(len(glob.glob(eval_dir+'/*'))):
#     classes.append(glob.glob(eval_dir+'/*')[i].split('/')[-1])
# class_num = len(classes)
# print('Total Class num: ',class_num)
# print('Class label:')
# classes=sorted(classes)
#
# for i in range(class_num):print('{:5d}th : ' .format(i+1),classes[i])



# 데이터 전처리

val_transforms = transforms.Compose([
                                       transforms.Resize((600,380)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])


def roc_curve_plot(y_test , pred_proba_c1):
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
    plt.savefig(os.path.join(save_dir, f"best_rocAP_Test_chuncheon600.png"),dpi=2000)
    plt.close()

def roc_curve_plot_multi(y_test_list, pred_proba_c1_list, model_labels, save_dir):
    plt.figure(figsize=(10, 8))

    for y_test, pred_proba_c1, label in zip(y_test_list, pred_proba_c1_list, model_labels):
        fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
        plt.plot(fprs, tprs, label=label)

    plt.plot([0, 1], [0, 1], 'k--', label='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR( 1 - Sensitivity )', fontsize=20)  # Doubled the font size for xlabel
    plt.ylabel('TPR( Recall )', fontsize=20)  # Doubled the font size for ylabel
    plt.legend(fontsize='large')  # Increased the font size for legend
    plt.savefig(os.path.join(save_dir, "multi_model_roc_curve.png"), dpi=2000)
    plt.close()


# 학습 함수


# 검증 함수
def validate(model, val_loader, device):
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
            y_true += labels.cpu().numpy().tolist()
            y_pred += (outputs.sigmoid().cpu().detach().numpy() > 0.5).astype(int).tolist()
            y_prob += (outputs.sigmoid().cpu().detach().numpy()).tolist()

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)

    return  acc, auc, f1, y_true, y_prob

def validate_ens(model, val_loader, device):
    for i in model:
        i.eval()

    running_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(val_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs1 = model[0](inputs)
            outputs2 = model[1](inputs)
            outputs3 = model[2](inputs)
            outputs4 = model[3](inputs)
            outputs5 = model[4](inputs)
            outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5)/5
            y_true += labels.cpu().numpy().tolist()
            y_pred += (outputs.sigmoid().cpu().detach().numpy() > 0.5).astype(int).tolist()
            y_prob += (outputs.sigmoid().cpu().detach().numpy()).tolist()
            # print(y_prob)

    # acc = accuracy_score(y_true, y_pred)
    # auc = roc_auc_score(y_true, y_prob)
    # f1 = f1_score(y_true, y_pred)
    # roc_curve_plot_multi(y_true, y_prob)
    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(5, 5))
    # sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.title('Confusion Matrix', size=15)
    # plt.savefig(os.path.join(save_dir,'confusion_matrix_AP_chuncheon.png'),dpi = 2000 ) # save the figure to file
    return y_true, y_prob
    # return acc, auc, f1, y_true, y_prob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# weight_path = 'model_effAP0.pt'
weight_path1 = 'model_finaldata_effAP6000.pt'
weight_path2 = 'model_finaldata_effAP6001.pt'
weight_path3 = 'model_finaldata_effAP6002.pt'
weight_path4 = 'model_finaldata_effAP6003.pt'
weight_path5 = 'model_finaldata_effAP6004.pt'

weight_path1_LAT = 'model_finaldata_effLAT6000.pt'
weight_path2_LAT = 'model_finaldata_effLAT6001.pt'
weight_path3_LAT = 'model_finaldata_effLAT6002.pt'
weight_path4_LAT = 'model_finaldata_effLAT6003.pt'
weight_path5_LAT = 'model_finaldata_effLAT6004.pt'

# weight_path1 = 'model_finaldata_effLAT6000.pt'
# weight_path2 = 'model_finaldata_effLAT6001.pt'
# weight_path3 = 'model_finaldata_effLAT6002.pt'
# weight_path4 = 'model_finaldata_effLAT6003.pt'
# weight_path5 = 'model_finaldata_effLAT6004.pt'

num_classes = 1
batch_size = 1

eval_dir_AP = 'dongtan_validation_AP/'
eval_dir_LAT = 'dongtan/'
test_data_AP = datasets.ImageFolder(eval_dir_AP, transform=val_transforms)
test_loader_AP = DataLoader(dataset=test_data_AP, batch_size=batch_size, shuffle=False)

test_data_LAT = datasets.ImageFolder(eval_dir_LAT, transform=val_transforms)
test_loader_LAT = DataLoader(dataset=test_data_LAT, batch_size=batch_size, shuffle=False)

# # 0번 클래스의 인덱스만 추출
class_0_indices_AP = [i for i, (_, label) in enumerate(test_data_AP.samples) if label == 0]
test_loader_AP_class_0 = DataLoader(dataset=Subset(test_data_AP, class_0_indices_AP), batch_size=batch_size, shuffle=False)

class_1_indices_AP = [i for i, (_, label) in enumerate(test_data_AP.samples) if label == 1]
test_loader_AP_class_1 = DataLoader(dataset=Subset(test_data_AP, class_1_indices_AP), batch_size=batch_size, shuffle=False)

test_data_LAT = datasets.ImageFolder(eval_dir_LAT, transform=val_transforms)
test_loader_LAT = DataLoader(dataset=test_data_LAT, batch_size=batch_size, shuffle=False)

class_0_indices_LAT = [i for i, (_, label) in enumerate(test_data_LAT.samples) if label == 0]
test_loader_LAT_class_0 = DataLoader(dataset=Subset(test_data_LAT, class_0_indices_LAT), batch_size=batch_size, shuffle=False)

class_1_indices_LAT = [i for i, (_, label) in enumerate(test_data_LAT.samples) if label == 1]
test_loader_LAT_class_1 = DataLoader(dataset=Subset(test_data_LAT, class_1_indices_LAT), batch_size=batch_size, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
# 모델 불러오기
model1 = models.efficientnet_v2_s(pretrained=True)
model1.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model1.load_state_dict(torch.load(weight_path1, map_location=device))
model1 = model1.to(device)

model2 = models.efficientnet_v2_s(pretrained=True)
model2.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model2.load_state_dict(torch.load(weight_path2, map_location=device))
model2 = model2.to(device)

model3 = models.efficientnet_v2_s(pretrained=True)
model3.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model3.load_state_dict(torch.load(weight_path3, map_location=device))
model3 = model3.to(device)

model4 = models.efficientnet_v2_s(pretrained=True)
model4.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model4.load_state_dict(torch.load(weight_path4, map_location=device))
model4 = model4.to(device)

model5 = models.efficientnet_v2_s(pretrained=True)
model5.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model5.load_state_dict(torch.load(weight_path5, map_location=device))
model5 = model5.to(device)

model_AP = [model1,model2,model3,model4,model5]

model1_LAT = models.efficientnet_v2_s(pretrained=True)
model1_LAT.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model1_LAT.load_state_dict(torch.load(weight_path1_LAT, map_location=device))
model1_LAT = model1_LAT.to(device)

model2_LAT = models.efficientnet_v2_s(pretrained=True)
model2_LAT.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model2_LAT.load_state_dict(torch.load(weight_path2_LAT, map_location=device))
model2_LAT = model2_LAT.to(device)

model3_LAT = models.efficientnet_v2_s(pretrained=True)
model3_LAT.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model3_LAT.load_state_dict(torch.load(weight_path3_LAT, map_location=device))
model3_LAT = model3_LAT.to(device)

model4_LAT = models.efficientnet_v2_s(pretrained=True)
model4_LAT.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model4_LAT.load_state_dict(torch.load(weight_path4_LAT, map_location=device))
model4_LAT = model4_LAT.to(device)

model5_LAT = models.efficientnet_v2_s(pretrained=True)
model5_LAT.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3,inplace=True),
                                           torch.nn.Linear(in_features=1280, out_features=1,bias= True))
model5_LAT.load_state_dict(torch.load(weight_path5_LAT, map_location=device))
model5_LAT = model5_LAT.to(device)

model_LAT = [model1_LAT,model2_LAT,model3_LAT,model4_LAT,model5_LAT]


y_true_AP, y_prob_AP_class_0 = validate_ens(model_AP,test_loader_AP_class_0, device)
# print(np.mean(np.array(y_prob_AP_class_0)))
y_true_AP, y_prob_AP_class_1 = validate_ens(model_AP,test_loader_AP_class_1, device)
y_true_AP, y_prob_LAT_class_0 = validate_ens(model_LAT,test_loader_LAT_class_0, device)
y_true_AP, y_prob_LAT_class_1 = validate_ens(model_LAT,test_loader_LAT_class_1, device)
# # y_prob_list = [y_prob_AP, y_prob_LAT]
# # y_true_list = [y_true_AP, y_true_LAT]
# print(f'prob_AP_class_0 : {np.mean(np.array(y_prob_AP_class_0))}')
# print(f'prob_AP_class_1 : {np.mean(np.array(y_prob_AP_class_1))}')
# print(f'prob_LAT_class_0 : {np.mean(np.array(y_prob_LAT_class_0))}')
# print(f'prob_LAT_class_1 : {np.mean(np.array(y_prob_LAT_class_1))}')

y_prob_AP_class_0 = [math.floor((1-x[0]) * 10000) / 10000 for x in y_prob_AP_class_0]
y_prob_AP_class_1 = [math.floor(x[0] * 10000) / 10000 for x in y_prob_AP_class_1]
y_prob_LAT_class_0 = [math.floor((1-x[0]) * 10000) / 10000 for x in y_prob_LAT_class_0]
y_prob_LAT_class_1 = [math.floor(x[0] * 10000) / 10000 for x in y_prob_LAT_class_1]

img_numbers = list(range(52))

# 확률 값을 DataFrame으로 변환
df_probs = pd.DataFrame({
    'img': img_numbers,
    'prob_AP_0': y_prob_AP_class_0,
    'prob_LAT_0': y_prob_LAT_class_0,
    'prob_AP_1': y_prob_AP_class_1,
    'prob_LAT_1': y_prob_LAT_class_1
})

# CSV 파일로 저장
df_probs.to_csv('probabilities.csv', index=False)
print("end")

# acc, auc, f1, y_true_AP, y_prob_AP = validate_ens(model_AP,test_loader_AP, device)
# val_acc, val_auc, val_f1, y_true_LAT, y_prob_LAT = validate_ens(model_LAT,test_loader_LAT, device)
# y_prob_list = [y_prob_AP, y_prob_LAT]
# y_true_list = [y_true_AP, y_true_LAT]

# roc_curve_plot_multi(y_true_list,y_prob_list, model_labels=['AP','LAT'],save_dir=save_dir)
# print(f'Val Acc: {val_acc:.4f},  Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}')
# roc_curve_plot(y_true, y_prob)


