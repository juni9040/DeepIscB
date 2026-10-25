
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

from Bio.SeqUtils import MeltingTemp as mt
from Bio.SeqUtils import GC as gc
from Bio.Seq import Seq
from RNA import fold_compound
import re


input_file = 'example_input.csv'
output_file = 'example_output.csv'



#####################################################################################################################
#####################################################################################################################

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


def reverse_complement(sSeq):
    dict_sBases = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', 'U': 'U', 'n': '',
                   '.': '.', '*': '*', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}
    list_sSeq = list(sSeq)  # Turns the sequence in to a gigantic list
    list_sSeq = [dict_sBases[sBase] for sBase in list_sSeq]
    return ''.join(list_sSeq)[::-1]


def data_feature(data):

    data['guide_nGC1'] = data['context'].apply(lambda x:x[4:20].count('G')+x[4:20].count('C'))
    data['guide_fGC1'] = data['context'].apply(lambda x:gc(x[4:20]))
    data['guide_nGC2'] = data['context'].apply(lambda x:x[20:26].count('G')+x[20:26].count('C'))
    data['guide_fGC2'] = data['context'].apply(lambda x:gc(x[20:26]))
    data['guide_nGC3'] = data['context'].apply(lambda x:x[4:26].count('G')+x[4:26].count('C'))
    data['guide_fGC3'] = data['context'].apply(lambda x:gc(x[4:26]))
    data['guide_polN'] = data['context'].apply(lambda x:len(re.findall(r'(\w)\1+', x[4:20])))
    data['guide_polAT'] = data['context'].apply(lambda x:len(re.findall(r'([AT])\1+', x[4:20])))
    data['guide_polGC'] = data['context'].apply(lambda x:len(re.findall(r'([GC])\1+', x[4:20])))
    data['guide_TM'] = data['context'].apply(lambda x:mt.Tm_NN(seq = Seq(reverse_complement(x.upper().replace('A', 'U'))), nn_table=mt.R_DNA_NN1))
    data['guide_MFE'] = data['context'].apply(lambda x:fold_compound('G'+x[4:20]).mfe()[1])
    data['guide+s_MFE'] = data['context'].apply(lambda x:fold_compound('G'+x[4:20]+'GGCTCGTCCAACTGCGGTTGAACGAGCACAGGCTGAGACATTCGTAAGGCCGAAAGGCCGGACGCACCCTGGGATTTCCCCAGTCCCCGGAACTGCATAGCGGATGCCAGTTGATGGAGCAATCTATCAGATAAGCCAGGGGGAACAATCACCTCTCTGTATCAGAGAGAGTTTTACAAAAGGAGGAACGGTTTTTTT').mfe()[1])
    
    return data

data = pd.read_csv(input_file)
data = data_feature(data)

train_set = data
train_feature = train_set.iloc[:,1:14].astype('float32')
test_set = data
test_feature = test_set.iloc[:,1:14].astype('float32')
train_set['en'] = 0
train_set['enO'] = 0
train_set['LFC'] = 0
test_set['en'] = 0
test_set['enO'] = 0
test_set['LFC'] = 0

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
train_feature = torch.tensor(train_feature.values)
test_list = np.array(test_list)
test_vector = test_list==base_array
test_vector = torch.from_numpy(test_vector.transpose((1,0,2))*1).float()
test_feature = torch.tensor(test_feature.values)


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
        return output_en, output_enO, output_lfc

def final_prediction(fold, feature_dim, hidden_dim, drop):
    final_pred = []
    final_true = []
    for i in range(fold):
        model = DeepIscB(in_channels=4, feature_dim=feature_dim, hidden_dim=hidden_dim, drop=drop).to(device)
        model.load_state_dict(torch.load(f'model/model_20261024_feature_fix_FM1test_{i}.h5'))    
        model.eval()
        final_dataset = Dataset(test_vector, test_ans, test_feature)
        final_loader = DataLoader(final_dataset, batch_size=20000)
        
        with torch.no_grad():
            for input, result, feature in final_loader:
                input.to(device)
                pred_en, pred_enO, pred_lfc = model(input, feature)

                result_en = result[:,0].unsqueeze(1).cpu()
                result_enO = result[:,1].unsqueeze(1).cpu()
                result_lfc = result[:,2].unsqueeze(1).cpu()
                
                pred_en = pred_en.cpu()
                pred_enO = pred_enO.cpu()
                pred_lfc = pred_lfc.cpu()
                final_pred.append(torch.cat([pred_en, pred_enO, pred_lfc], dim=1))
                final_true.append(torch.cat([result_en, result_enO, result_lfc], dim=1))


    y_pred = torch.stack(final_pred, dim=2)
    y_true = torch.stack(final_true, dim=2)
    
    y_pred = torch.Tensor(y_pred).mean(dim=2)
    y_true = torch.Tensor(y_true).mean(dim=2)
    
    output_df1 = pd.DataFrame(list(test_set['context']), columns=['context'])
    output_df2 = pd.DataFrame(y_pred, columns=['enIscB_score','enOgeuIscB_score','lfc_pred']).drop(['lfc_pred'], axis=1)

    pd.concat([output_df1,output_df2], axis=1).to_csv(output_file, index=False)


final_prediction(fold=5, feature_dim=128, hidden_dim=128, drop=0.3)
