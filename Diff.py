import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch.optim as optim
import rms_norm import RMSNorm

import Initializer
import Data_reader
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import numpy as np

import pandas as pd
import matplotlib as plt
import seaborn as sns

class DifferentialAttention(nn.Module):
    def __init__(self, hidden_dim, lambda_init=0.8):
        super(DifferentialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.lambda_init = lambda_init
        self.lambda_q1 = nn.Parameter(torch.randn(hidden_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(hidden_dim))
        self.lambda_q2 = nn.Parameter(torch.randn(hidden_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, query1, query2, key1, key2, value):
        attn_scores1 = torch.matmul(query1, key1.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        attn_scores2 = torch.matmul(query2, key2.transpose(-2, -1)) / (self.hidden_dim ** 0.5)

        attn_probs1 = F.softmax(attn_scores1, dim=-1)
        attn_probs2 = F.softmax(attn_scores2, dim=-1)

        lambda_ = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1)) - torch.exp(
            torch.dot(self.lambda_q2, self.lambda_k2)) + self.lambda_init
        diff_attn_probs = attn_probs1 - lambda_ * attn_probs2

        output = torch.matmul(diff_attn_probs, value)
        return output

class DiffMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, lambda_init=0.8):
        super(DiffMultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.lambda_init = lambda_init

        self.query_linear = nn.Linear(input_dim, 2 * hidden_dim * num_heads)
        self.key_linear = nn.Linear(input_dim, 2 * hidden_dim * num_heads)
        self.value_linear = nn.Linear(input_dim, hidden_dim * num_heads)
        self.out_linear = nn.Linear(hidden_dim * num_heads, output_dim)
        self.heads = nn.ModuleList(
            [DifferentialAttention(hidden_dim, lambda_init) for _ in range(num_heads)])

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    def forward(self, data):
        x = data.x
        x = x.float()
        batch = data.batch
        batch_size = batch.max().item() + 1

        node_features = []
        for i in range(batch_size):
            mask = batch == i
            node_features.append(x[mask])

        outputs = []
        self.attention_scores = 
        for features in node_features:
            num_nodes = features.size(0)
            input_dim = features.size(1)

            q = self.query_linear(features).view(1, num_nodes, 2 * self.num_heads, self.hidden_dim).transpose(1, 2)
            k = self.key_linear(features).view(1, num_nodes, 2 * self.num_heads, self.hidden_dim).transpose(1, 2)
            v = self.value_linear(features).view(1, num_nodes, self.num_heads, self.hidden_dim).transpose(1, 2)

            head_outputs = []
            batch_attention_scores = []
            for i in range(self.num_heads):
                q1 = q[:, i, :, :]
                q2 = q[:, i + self.num_heads, :, :]
                k1 = k[:, i, :, :]
                k2 = k[:, i + self.num_heads, :, :]
                head_value = value[:, i, :, :]

                attn_scores1 = torch.matmul(q1, k1.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
                attn_scores2 = torch.matmul(q2, k2.transpose(-2, -1)) / (self.hidden_dim ** 0.5)

                attn_probs1 = F.softmax(attn_scores1, dim=-1)
                attn_probs2 = F.softmax(attn_scores2, dim=-1)

                lambda_ = torch.exp(torch.dot(self.heads[i].lambda_q1, self.heads[i].lambda_k1)) - torch.exp(
                    torch.dot(self.heads[i].lambda_q2, self.heads[i].lambda_k2)) + self.heads[i].lambda_init
                diff_attn_probs = attn_probs1 - lambda_ * attn_probs2

                batch_attention_scores.append(diff_attn_probs.squeeze().detach().cpu().numpy())

                head_output = self.heads[i](query1, query2, key1, key2, head_value)
                head_output = (1 - self.lambda_init) * F.normalize(head_output, p=2, dim=-1)
                head_outputs.append(head_output)

            self.attention_scores.append(np.stack(batch_attention_scores, axis=0))

            concat_output = torch.cat(head_outputs, dim=-1)
            output = self.out_linear(concat_output)
            output = self.subln(output)
            outputs.append(output)

        max_num_nodes = max([o.size(1) for o in outputs])
        padded_outputs = []
        for output in outputs:
            num_nodes = output.size(1)
            if num_nodes < max_num_nodes:
                padding = torch.zeros(1, max_num_nodes - num_nodes, output.size(2), device=output.device)
                output = torch.cat([output, padding], dim=1)
            padded_outputs.append(output)

        output = torch.cat(padded_outputs, dim=0)
        return output


hidden_dim = 32
output_dim = 64
num_heads = 8

class CombinedModel(nn.Module):
    def __init__(self, input_dim_1, input_dim_2):
        super(CombinedModel, self).__init__()
        self.model_1 = DiffMultiHeadAttention(input_dim_1, hidden_dim, output_dim, num_heads)
        self.model_2 = DiffMultiHeadAttention(input_dim_2, hidden_dim, output_dim, num_heads)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(output_dim * 2, 1)

    def forward(self, data_1, data_2):
        output_1 = self.model_1(data_1)
        output_2 = self.model_2(data_2)

        output_1 = torch.mean(output_1, dim=1)
        output_2 = torch.mean(output_2, dim=1)

        if output_1.dim() == 1:
            output_1 = output_1.unsqueeze(1)
        if output_2.dim() == 1:
            output_2 = output_2.unsqueeze(1)

        attention_output = torch.cat((output_1, output_2), dim=1)

        output = self.fc(attention_output)

        output = self.dropout(output)

        output = torch.sigmoid(output)
        return output

class CustomDataset(Dataset):
    def __init__(self, smiles_graph_1_list, smiles_graph_2_list, label_list):
        self.smiles_graph_1_list = smiles_graph_1_list
        self.smiles_graph_2_list = smiles_graph_2_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        return self.smiles_graph_1_list[idx], self.smiles_graph_2_list[idx], self.label_list[idx]

def custom_collate(batch):
    data_1_list, data_2_list, label_list = zip(*batch)
    data_1_batch = Batch.from_data_list(data_1_list)
    data_2_batch = Batch.from_data_list(data_2_list)
    label_batch = torch.tensor(label_list, dtype=torch.float32).unsqueeze(1)
    return data_1_batch, data_2_batch, label_batch

smiles_graph_1_list = []
smiles_graph_2_list = []
label_list = []
for n in range(1400):
    smiles_1, smiles_2, label = Data_reader.load_smiles(n)

    smiles_graph_1 = Initializer.graph(smiles_1)
    smiles_graph_2 = Initializer.graph(smiles_2)

    # p = 0

    # smiles_graph_1 = Initializer.remove_edges(smiles_graph_1,p)
    # smiles_graph_2 = Initializer.remove_edges(smiles_graph_2,p)

    smiles_graph_1_list.append(smiles_graph_1)
    smiles_graph_2_list.append(smiles_graph_2)
    label_list.append(label)

input_dim_1 = smiles_graph_1_list[0].x.size(1)
input_dim_2 = smiles_graph_2_list[0].x.size(1)

model = CombinedModel(input_dim_1, input_dim_2)

# model.load_state_dict(torch.load('trained_model_params.pth'))

if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
else:
    print('GPU Failed, buy a H100 instead!')
    device = torch.device("cpu")
    model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

dataset = CustomDataset(smiles_graph_1_list, smiles_graph_2_list, label_list)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

num_epochs = 50
epoch_list = []
ave_loss_list = []

for epoch in range(num_epochs):
    total_loss = 0
    for data_1_batch, data_2_batch, label_batch in dataloader:
        optimizer.zero_grad()
        data_1_batch = data_1_batch.to(device)
        data_2_batch = data_2_batch.to(device)
        label_batch = label_batch.to(device)
        output = model(data_1_batch, data_2_batch)
        assert output.size() == label_batch.size(), f"Output size {output.size()} and label size {label_batch.size()} do not match."
        loss = criterion(output, label_batch)
        total_loss += loss.item() * len(data_1_batch)
        loss.backward()
        optimizer.step()

    scheduler.step()

    ave_loss = total_loss / len(dataset)
    epoch_list.append(epoch)
    ave_loss_list.append(ave_loss)
    if epoch % 1 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}, Loss: {ave_loss}, Current LR: {current_lr}")

data = pd.DataFrame({'epoch': epoch_list, 'ave_loss': ave_loss_list})
plt.rcParams['figure.dpi'] = 300
sns.regplot(data=data, x='epoch', y='ave_loss', order=2)
plt.title('Epoch vs Average Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.savefig('epoch_vs_loss_neo.png')

torch.save(model.state_dict(), 'params.pth')
