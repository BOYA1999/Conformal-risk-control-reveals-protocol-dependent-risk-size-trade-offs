from enum import IntEnum

import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool


HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRALITIES = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]
NODE_DIM = 7 + len(HYBRIDIZATIONS) + 1 + len(CHIRALITIES) + 1
EDGE_DIM = len(BOND_TYPES) + 1 + 2 + len(BOND_STEREOS) + 1


def one_hot(value, choices):
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_features(atom):
    return [
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalDegree() / 6.0,
        atom.GetFormalCharge() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        atom.GetMass() / 200.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        *one_hot(atom.GetHybridization(), HYBRIDIZATIONS),
        *one_hot(atom.GetChiralTag(), CHIRALITIES),
    ]


def bond_features(bond):
    return [
        *one_hot(bond.GetBondType(), BOND_TYPES),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *one_hot(bond.GetStereo(), BOND_STEREOS),
    ]


def graph_from_smiles(smiles, activity=0.0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"invalid molecular graph: {smiles}")
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edges = []
    attributes = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = bond_features(bond)
        edges.extend([(a, b), (b, a)])
        attributes.extend([feature, feature])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([activity], dtype=torch.float32))


class GINERegressor(nn.Module):
    def __init__(self, hidden=96, layers=4, dropout=0.1):
        super().__init__()
        self.node_projection = nn.Linear(NODE_DIM, hidden)
        self.convolutions = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))
            self.convolutions.append(GINEConv(mlp, train_eps=True, edge_dim=EDGE_DIM))
            self.normalizations.append(nn.LayerNorm(hidden))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        hidden = self.node_projection(x)
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            update = convolution(hidden, edge_index, edge_attr)
            hidden = normalization(hidden + self.dropout(update))
        return self.readout(global_mean_pool(hidden, batch)).squeeze(-1)
