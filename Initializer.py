import torch
import torch_geometric
from torch_geometric.data import Data
from rdkit import Chem
import Data_reader

#读取原子信息作为边的特征.
#这个东西输出的PyG.Data.x的形状是[Num_nodes, 78].

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return torch.tensor(list(map(lambda s: x == s, allowable_set)), dtype=torch.float)


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return torch.tensor(list(map(lambda s: x == s, allowable_set)), dtype=torch.float)


def atom_features(atom):
    symbol_encoding = one_of_k_encoding_unk(atom.GetSymbol(),
                                    ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                     'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                     'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                     'Pt', 'Hg', 'Pb', 'Unknown'])
    degree_encoding = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    total_hs_encoding = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    implicit_valence_encoding = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    is_aromatic = torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
    return torch.cat((symbol_encoding, degree_encoding, total_hs_encoding, implicit_valence_encoding, is_aromatic))
#特征包括：原子是什么，它的度（相连的原子数量），周围有几个氢，化合价，是否有芳香性。


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    num_nodes = mol.GetNumAtoms()
    node_features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        node_features.append(feature)
    node_features = torch.stack(node_features)
    edge_index = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        #添加双向边
        edge_index.extend([[start, end], [end, start]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    graph = Data(x=node_features, edge_index=edge_index)
    return graph


def graph(smiles):
    return smiles_to_graph(smiles)

n = 0

smiles_1, smiles_2, label = Data_reader.load_smiles(n)

Graph_1 = graph(smiles_1)
Graph_2 = graph(smiles_2)

if torch.cuda.is_available():
    device = torch.device("cuda")

Graph_1 = Graph_1.to(device)
Graph_2 = Graph_2.to(device)

def remove_edges(data, p):
    edge_index = data.edge_index 
    if edge_index.size(1)  == 0:
        return data  # 空边直接返回
    
    u, v = edge_index
    sorted_u = torch.where(u  <= v, u, v)
    sorted_v = torch.where(u  <= v, v, u)
    sorted_edges = torch.stack([sorted_u,  sorted_v], dim=0).t()
    
    unique_edges = torch.unique(sorted_edges,  dim=0)
    num_unique = unique_edges.size(0) 
    num_remove = int(num_unique * p)
    
    if num_remove == 0:
        return data
    
    remove_idx = torch.randperm(num_unique)[:num_remove] 
    edges_to_remove = unique_edges[remove_idx]
    
    edges_to_remove_set = set(map(tuple, edges_to_remove.cpu().numpy())) 
    mask = []
    for edge in sorted_edges.cpu().numpy(): 
        mask.append(tuple(edge)  not in edges_to_remove_set)
    mask = torch.tensor(mask,  dtype=torch.bool,  device=edge_index.device) 
    new_data = Data()
    new_data.edge_index  = edge_index[:, mask]
    
    for key in data.keys():
        if key == 'edge_index':
            continue
        if data.is_edge_attr(key): 
            new_data[key] = data[key][mask]
        else:
            new_data[key] = data[key]
    
    return new_data

def remove_nodes(data, p):
    num_nodes = data.x.size(0)
    num_remove = int(num_nodes * p)
    if num_remove == 0:
        return data
    remove_indices = torch.randperm(num_nodes)[:num_remove]
    keep_indices = ~torch.isin(torch.arange(num_nodes), remove_indices)

    new_x = data.x[keep_indices]

    new_edge_index = []
    for edge in data.edge_index.t():
        source, target = edge
        if keep_indices[source] and keep_indices[target]:
            new_source = torch.sum(keep_indices[:source])
            new_target = torch.sum(keep_indices[:target])
            new_edge_index.append([new_source, new_target])
    new_edge_index = torch.tensor(new_edge_index, dtype = torch.long).t().contiguous()

    new_data = Data(x = new_x, edge_index = new_edge_index)
    return new_data

if __name__ == "__main__":
    for epoch in range(3):
        n = epoch
        smiles_1, smiles_2, label = Data_reader.load_smiles(n)

        Graph_1 = graph(smiles_1)
        Graph_2 = graph(smiles_2)

        graph_1 = smiles_to_graph(smiles_1)
        graph_2 = smiles_to_graph(smiles_2)

        graph_1_1 = remove_nodes(graph_1,0.3)

        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0).to(device)
        print(graph_1, graph_1_1, label)
