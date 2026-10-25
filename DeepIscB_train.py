import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Concatenate
from keras.layers import Dense, Dropout, Activation, Flatten, Multiply
from keras.layers import Conv1D, AveragePooling1D
import numpy as np
import pandas as pd
from sklearn import metrics
import sklearn
import random
import os
import time
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import torch.utils.data
import scipy.stats as stats
from tqdm import tqdm

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
seed_everything(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

#run setting : early stopping으로 epoch 대체
RUN_TAG = '20261024_feature_fix_FM1test'
MODEL_PREFIX = f'model_{RUN_TAG}'
STUDY_DB = f'optuna_{RUN_TAG}.db'
MAX_EPOCHS = int(os.environ.get('MAX_EPOCHS', '500'))   #early stopping 상한
PATIENCE = int(os.environ.get('PATIENCE', '15'))        #validation loss 개선 없이 버티는 epoch 수
MIN_DELTA = 1e-6      #개선으로 인정하는 최소 변화량
N_TRIALS = int(os.environ.get('N_TRIALS', '0')) or None

#data preprocessing
def data_preprocessing(data):
    print(data)
    print('Total data: ', len(data))
    #data = data[data['Lib_Type'].isin(['FM_1st_Lib', 'FM_2nd_Lib', 'FM_3rd_Lib', 'TAM_Lib'])]
    #data = data[data['Total']>100]
    #data = data[data['Total_ctrl']>100]
    print('Total data: ', len(data))
    #data = data.iloc[:,[2,9]]
    data.columns = ['context', 'enIscB', 'enOIscB', 'LFC', 'class']
    data = data[['context', 'enIscB', 'enOIscB', 'LFC']]
    
    #subset = ['train']*16384 + ['test']*877
    #subset = ['test']*len(data)
    #TAM library 제외할 경우 2500, 1142 
    #test original = 1068

    return data



data = pd.read_csv('0521_IscB_input_FM1test.csv')
#data = data_preprocessing(data)
print(data.iloc[:,12:24])
#training set / test set 구분

train_set = data[data['Fold']!='test']
train_feature = train_set.iloc[:,12:24].astype('float32')
train_fold = train_set['Fold']
test_set = data[data['Fold']=='test']
test_feature = test_set.iloc[:,12:24].astype('float32')
#gpu에 dataset load, one-hot vector
'''
data = pd.read_csv('endo_repeat_input.csv')
print(data.iloc[:,2:14])
train_set = data
train_feature = train_set.iloc[:,2:14].astype('float32')
test_set = data
test_feature = test_set.iloc[:,2:14].astype('float32')
train_set['en'] = 0
train_set['enO'] = 0
train_set['LFC'] = 0
test_set['en'] = 0
test_set['enO'] = 0
test_set['LFC'] = 0
'''
train_list = []
for seq in train_set['context']:
    train_list.append(list(seq))

test_list = []
for seq in test_set['context']:
    test_list.append(list(seq))



train_list = np.array(train_list)
base_array = np.array([[['A']], [['T']], [['G']], [['C']]])
train_vector = train_list==base_array
train_vector = torch.from_numpy(train_vector.transpose((1,0,2))*1).float()
print(train_vector.shape)
train_feature = torch.tensor(train_feature.values)
print(train_feature.shape)
test_list = np.array(test_list)
test_vector = test_list==base_array
test_vector = torch.from_numpy(test_vector.transpose((1,0,2))*1).float()
print(test_vector.shape)
test_feature = torch.tensor(test_feature.values)
print(test_feature.shape)


train_ans = torch.Tensor(list(zip(list(train_set['en']),list(train_set['enO']),list(train_set['LFC']))))
test_ans = torch.Tensor(list(zip(list(test_set['en']),list(test_set['enO']),list(test_set['LFC']))))


class Dataset(Dataset):
    def __init__(self, data, labels, feature):
        self.data = data.to(device)
        self.labels = labels.to(device)
        self.feature = feature.to(device)
    def __len__(self):
        return (len(self.data))
    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.feature[index]

#CNN model-inception
class DeepIscB(nn.Module):
    def __init__(self, in_channels=4, seq_len=30, feature_dim=128, hidden_dim=512, drop=0.3):
        super(DeepIscB, self).__init__()
        self.seq_len = seq_len

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=30, kernel_size=1, padding='same')
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=100, kernel_size=3, padding='same')
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=40, kernel_size=5, padding='same')
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=30, kernel_size=7, padding='same')
        )
        self.feature_layer = nn.Sequential(
            nn.Linear(12,feature_dim),
            nn.ELU(),
            nn.Dropout(drop)
        )
        self.flatten = nn.Flatten()
        decoder_input_dim = 200*self.seq_len+feature_dim

        def create_decoder():
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(decoder_input_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim,128),
                nn.ELU(),
                nn.Dropout(drop),
                nn.Linear(128,1)
            )
        self.decoder1 = create_decoder()
        self.decoder2 = create_decoder()
        self.decoder3 = create_decoder()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1,128),
            nn.ELU(),
            nn.Linear(128,64),
            nn.ELU(),
            nn.Linear(64,2)
        )
    def forward(self, x, feature):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        output = torch.cat([out1, out2, out3, out4],1)
        output = self.flatten(output)
        feature_out = self.feature_layer(feature)
        output = torch.cat([output, feature_out],1)

        output_en = self.decoder1(output)
        output_enO = self.decoder2(output)
        output_lfc = self.decoder3(output)
        #class_input = torch.cat([output_en, output_enO, output_lfc],1)
        #output_class = self.classifier(output_lfc)
        return output_en, output_enO, output_lfc


#validation fold loss (en+enO), early stopping 기준
def evaluate_loss(model, loader):
    criterion = nn.MSELoss(reduction='mean')
    model.eval()
    total_loss = 0.0
    n_sample = 0
    with torch.no_grad():
        for input, result, feature in loader:
            input = input.to(device)
            result = result.to(device)
            feature = feature.to(device)

            pred_en, pred_enO, pred_lfc = model(input, feature)
            loss = criterion(pred_en, result[:,0].unsqueeze(1)) + criterion(pred_enO, result[:,1].unsqueeze(1))
            total_loss += loss.item()*input.size(0)
            n_sample += input.size(0)
    return total_loss/max(n_sample,1)


def train_model(train_loader, test_loader, lr, feature_dim, hidden_dim, drop,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, min_delta=MIN_DELTA, verbose=False):
    model = DeepIscB(in_channels=4, feature_dim=feature_dim, hidden_dim=hidden_dim, drop=drop).to(device)
    #print(model)

    criterion_en = nn.MSELoss(reduction='mean')
    criterion_enO = nn.MSELoss(reduction='mean')
    criterion_lfc = nn.MSELoss(reduction='mean')
    criterion_class = nn.CrossEntropyLoss(reduction='mean')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_state = None
    best_epoch = 0
    wait = 0
    epoch = 0
    t_start = time.time()
    while epoch<max_epochs:
        model.train()
        total_loss = 0.0
        for input, result, feature in train_loader:
            input = input.to(device)
            result = result.to(device)
            feature = feature.to(device)

            result_en = result[:,0].unsqueeze(1)
            result_enO = result[:,1].unsqueeze(1)
            result_lfc = result[:,2].unsqueeze(1)
            #result_class = result[:,3].unsqueeze(1)
            #result_class = result_class.squeeze(dim=-1).type(torch.LongTensor).to(device)
            #print(result_en, result_enO, result_lfc)
            optimizer.zero_grad()
            pred_en, pred_enO, pred_lfc = model(input, feature)
            #print(torch.cat((result_class, pred_class),dim=1))
            loss_en = criterion_en(pred_en, result_en)
            loss_enO = criterion_enO(pred_enO, result_enO)
            loss_lfc = criterion_lfc(pred_lfc, result_lfc)
            #loss_class = criterion_class(pred_class, result_class)
            loss_total = loss_en+loss_enO
            

            loss_total.backward()
            optimizer.step()

            total_loss += loss_total.item()
        epoch+=1

        #early stopping : validation fold loss 기준으로 best weight 보관
        val_loss = evaluate_loss(model, test_loader)
        if verbose:
            print(f'Epoch [{epoch} / {max_epochs}] train loss : {total_loss/len(train_loader):.6f} | val loss : {val_loss:.6f}')
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    train_time = time.time()-t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    stopped = 'early stop' if wait >= patience else 'max epoch'
    print(f'  {stopped} : ran {epoch} epochs, best epoch {best_epoch} (val loss {best_loss:.6f}), '
          f'train time {train_time:.1f}s ({train_time/60:.2f}min)')

    model.eval()

    en_pred = []
    en_true = []

    enO_pred = []
    enO_true = []
    
    lfc_pred = []
    lfc_true = []

    class_pred = []
    class_true = []

    with torch.no_grad():
        for input, result, feature in test_loader:
            input.to(device)

            pred_en, pred_enO, pred_lfc = model(input, feature)

            result_en = result[:,0].unsqueeze(1).cpu()
            result_enO = result[:,1].unsqueeze(1).cpu()
            result_lfc = result[:,2].unsqueeze(1).cpu()
            #result_class = result[:,3].unsqueeze(1).cpu()
            pred_en = pred_en.cpu()
            pred_enO = pred_enO.cpu()
            pred_lfc = pred_lfc.cpu()
            #pred_class = pred_class.cpu()
            en_true.append(result_en.numpy())
            enO_true.append(result_enO.numpy())
            lfc_true.append(result_lfc.numpy())
            #class_true.append(result_class.numpy())
            en_pred.append(pred_en.numpy())
            enO_pred.append(pred_enO.numpy())
            lfc_pred.append(pred_lfc.numpy())
            #class_pred.append(pred_class.numpy())
    def validation(pred, true):
        y_pred = np.vstack(pred)
        y_true = np.vstack(true)
        print(len(y_pred),len(y_true))
        mse = mean_squared_error(y_true, y_pred)
        r2, pv = stats.pearsonr(y_true.squeeze(), y_pred.squeeze())
        spear, pv = stats.spearmanr(y_true, y_pred)
        print(f'MSE : {mse} |  pearson : {r2}    | spear : {spear}')
        return mse, r2, spear
    mse_en, r2_en, spear_en = validation(en_pred, en_true)
    mse_enO, r2_enO, spear_enO = validation(enO_pred, enO_true)
    mse_lfc, r2_lfc, spear_lfc = validation(lfc_pred, lfc_true)
    #mse_class, r2_class, spear_class = validation(class_pred, class_true)
    return (mse_en+mse_enO)/2, (r2_en+r2_enO)/2,(spear_en+spear_enO)/2,model,best_epoch,epoch,train_time

def kfold_training(train_dataset, train_fold, batch_size, lr, feature_dim, hidden_dim, drop,
                   save_prefix=None, verbose=False):
    seed_everything(42)
    kf = KFold(fold_split, shuffle=True)
    sum_mse = 0
    sum_r2 = 0
    sum_spear = 0
    fold_list = np.array(train_fold)
    fold_times = []
    fold_epochs = []
    final_pred = []
    final_true = []
    #for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
    for fold in range(5):
        #print(fold_list)
        train_idx = np.where(train_fold != str(fold))[0]
        test_idx = np.where(train_fold == str(fold))[0]
        #k fold cross validation training
        #print(train_idx, test_idx)
        print(f"fold {fold}\n-----------")

        train_loader = DataLoader(train_dataset, batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
        test_loader = DataLoader(train_dataset, batch_size, sampler=torch.utils.data.SubsetRandomSampler(test_idx))
        
        mse, r2, spear, model, best_epoch, ran_epoch, train_time = train_model(
            train_loader=train_loader, test_loader=test_loader, lr=lr,
            feature_dim=feature_dim, hidden_dim=hidden_dim, drop=drop, verbose=verbose)
        sum_mse += mse
        sum_r2 += r2
        sum_spear += spear
        fold_times.append(train_time)
        fold_epochs.append(best_epoch)
        print(f'fold {fold} : mse {mse:.6f} | pearson {r2:.4f} | spearman {spear:.4f} | '
              f'best epoch {best_epoch} | time {train_time:.1f}s')
        if save_prefix is not None:
            torch.save(model.state_dict(), f'{save_prefix}_{fold}.h5')

    print('MSE : ', mse)
    print("R2 : ", r2)
    print('spear : ', spear)

    print(f'avg mse : {sum_mse/5}, avg r2 : {sum_r2/5}, avg spearman : {sum_spear/5}')
    print(f'fold times (s) : {[round(t,1) for t in fold_times]} | total {sum(fold_times):.1f}s '
          f'({sum(fold_times)/60:.2f}min) | mean {np.mean(fold_times):.1f}s | best epochs {fold_epochs}')
    return sum_spear/5




train_dataset = Dataset(train_vector, train_ans, train_feature)
batch_size = 256
lr = 0.001
fold_split = 5

import optuna
def objective(trial):

    lr = trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 2e-4, 2e-3, 5e-4, 5e-3])
    batch_size = trial.suggest_categorical('batch_size', [32,64,128,256,512])
    #fold_split = trial.suggest_categorical('cross_validation_fold', [2,4,5,6,8,12])
    feature_dim = trial.suggest_categorical('feature_dimension', [64,128,256])
    hidden_dim = trial.suggest_categorical('hidden_layer_dimension', [128,256,512])
    drop = trial.suggest_categorical('dropout_value', [0.1,0.2,0.3])
    print(f'=== trial {trial.number} params : {trial.params} (early stopping : max_epochs={MAX_EPOCHS}, patience={PATIENCE})')
    t_trial = time.time()
    output = kfold_training(train_dataset, train_fold, batch_size, lr, feature_dim, hidden_dim, drop)
    print(f'=== trial {trial.number} done : spearman {output:.4f} | trial time {time.time()-t_trial:.1f}s '
          f'({(time.time()-t_trial)/60:.2f}min)', flush=True)

    return output


search_space = {
    'learning_rate' : [1e-4, 1e-3, 2e-4, 2e-3, 5e-4, 5e-3],
    'batch_size' : [32,64,128,256,512],
    'feature_dimension': [64,128,256],
    'hidden_layer_dimension': [128,256,512],
    'dropout_value': [0.1,0.2,0.3],
}
#GridSampler는 search_space가 바뀌면 기존 trial을 인식 못함 -> 미탐색 조합만 직접 enqueue
#skip_if_exists=True : COMPLETE/WAITING 등 DB에 이미 있는 조합은 건너뜀 (재실행해도 중복 없음)
study = optuna.create_study(direction='maximize', study_name='DeepIscB',
                            storage=f'sqlite:///{STUDY_DB}', load_if_exists=True,
                            sampler=optuna.samplers.RandomSampler(seed=42))
import itertools
for values in itertools.product(*search_space.values()):
    study.enqueue_trial(dict(zip(search_space.keys(), values)), skip_if_exists=True)
n_waiting = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.WAITING,)))
n_complete = len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)))
print(f'grid : total {len(list(itertools.product(*search_space.values())))} | complete {n_complete} | queued {n_waiting}')
t_study = time.time()
study.optimize(lambda trial:objective(trial), n_trials=min(N_TRIALS or n_waiting, n_waiting))
print(f'grid search total time : {time.time()-t_study:.1f}s ({(time.time()-t_study)/3600:.2f}h)')

print(study.best_trial, study.best_params, study.best_value)

#best parameter로 재학습 후 모델 저장
best = study.best_params
print('retraining with best params : ', best)
t_best = time.time()
kfold_training(train_dataset, train_fold,
               batch_size=best['batch_size'], lr=best['learning_rate'],
               feature_dim=best['feature_dimension'], hidden_dim=best['hidden_layer_dimension'],
               drop=best['dropout_value'], save_prefix=MODEL_PREFIX)
print(f'best-param retraining time : {time.time()-t_best:.1f}s ({(time.time()-t_best)/60:.2f}min)')
with open(f'{RUN_TAG}_params.json','w') as f:
    json.dump({'best_params': best, 'best_value': study.best_value,
               'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'min_delta': MIN_DELTA}, f, indent=2)

#kfold_training(train_dataset, train_fold, batch_size=128, lr=0.0005, feature_dim=128, hidden_dim=512, drop=0.3, save_prefix=MODEL_PREFIX)

#k fold cross validtion test output 

def final_prediction(fold, feature_dim, hidden_dim, drop):
    final_pred = []
    final_true = []
    for i in range(fold):
        model = DeepIscB(in_channels=4, feature_dim=feature_dim, hidden_dim=hidden_dim, drop=drop).to(device)
        model.load_state_dict(torch.load(f'{MODEL_PREFIX}_{i}.h5'))    
        model.eval()
        final_dataset = Dataset(test_vector, test_ans, test_feature)
        #final_dataset = Dataset(train_vector, train_ans)
        final_loader = DataLoader(final_dataset, batch_size=20000)
        
        with torch.no_grad():
            for input, result, feature in final_loader:
                input.to(device)
                pred_en, pred_enO, pred_lfc = model(input, feature)

                result_en = result[:,0].unsqueeze(1).cpu()
                result_enO = result[:,1].unsqueeze(1).cpu()
                result_lfc = result[:,2].unsqueeze(1).cpu()
                #result_class = result[:,3].unsqueeze(1).cpu()
                
                pred_en = pred_en.cpu()
                pred_enO = pred_enO.cpu()
                pred_lfc = pred_lfc.cpu()
                #pred_class = pred_class.cpu()
                final_pred.append(torch.cat([pred_en, pred_enO, pred_lfc], dim=1))
                final_true.append(torch.cat([result_en, result_enO, result_lfc], dim=1))


    y_pred = torch.stack(final_pred, dim=2)
    y_true = torch.stack(final_true, dim=2)
    
    y_pred = torch.Tensor(y_pred).mean(dim=2)
    y_true = torch.Tensor(y_true).mean(dim=2)
    
    output_df1 = pd.DataFrame(list(test_set['context']))
    output_df2 = pd.DataFrame(y_true, columns=['en_true','enO_true','lfc_true'])
    output_df3 = pd.DataFrame(y_pred, columns=['en_pred','enO_pred','lfc_pred'])
    pd.concat([output_df1,output_df2,output_df3], axis=1).to_csv(f"{RUN_TAG}_IscB_output.csv")

    mse = mean_squared_error(y_true, y_pred)
    #r2 = stats.pearsonr(y_true, y_pred)
    #spear, pv = stats.spearmanr(y_true, y_pred)
    print(mse)
    print(stats.pearsonr(output_df2['en_true'], output_df3['en_pred']),stats.pearsonr(output_df2['enO_true'], output_df3['enO_pred']),stats.pearsonr(output_df2['lfc_true'], output_df3['lfc_pred']))
    print(stats.spearmanr(output_df2['en_true'], output_df3['en_pred']),stats.spearmanr(output_df2['enO_true'], output_df3['enO_pred']),stats.spearmanr(output_df2['lfc_true'], output_df3['lfc_pred']))


t_pred = time.time()
final_prediction(fold=5, feature_dim=best['feature_dimension'],
                 hidden_dim=best['hidden_layer_dimension'], drop=best['dropout_value'])
print(f'final prediction time : {time.time()-t_pred:.1f}s')
