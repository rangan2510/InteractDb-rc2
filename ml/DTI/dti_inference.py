import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from einops import rearrange, repeat
import random
from rdkit import Chem
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from transformers import AutoModel, AutoTokenizer, pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
split = 0.8
batch_size = 64
epochs = 50
learning_rate = 1e-4
seed = 42
random.seed(seed)


df = pd.read_excel("drug-target-moa.xlsx")
df["action"].replace(
    [
        "blocker",
        "downregulator",
        "inactivator",
        "translocation inhibitor",
        "weak inhibitor",
    ],
    "inhibitor",
    inplace=True,
)
df["action"].replace(["binder", "binding", "substrate"], "ligand", inplace=True)
df["action"].replace(
    [
        "inhibitory allosteric modulator",
        "negative modulator",
        "positive allosteric modulator",
        "modulator",
    ],
    "allosteric modulator",
    inplace=True,
)
df["action"].replace(["partial antagonist"], "antagonist", inplace=True)
df["action"].replace(["inverse agonist", "partial agonist"], "agonist", inplace=True)
df["action"].replace(
    ["activator", "stimulator", "potentiator"], "inducer", inplace=True
)

df = df[df["uniprotkb-id"] != "Q8WXI7"]  # Too long
df = df[df["structure"].str.len() != 748]  # SMILE tokenizer sequence larger than 512

# df = df.loc[df['action'].isin(['inhibitor', 'ligand', 'allosteric modulator', 'antagonist', 'agonist', 'inducer'])]

df = df.drop_duplicates()
df = df.reset_index()
df.to_excel('dataset_processsed.xlsx')


class MyFeatureExtractionPipeline(FeatureExtractionPipeline):
    def preprocess(self, inputs):
        return_tensors = self.framework
        model_inputs = self.tokenizer(
            inputs, return_tensors=return_tensors, padding=True
        )
        return model_inputs


prot_model = MyFeatureExtractionPipeline(
    task="feature-extraction",
    model=AutoModel.from_pretrained("Rostlab/prot_bert"),
    tokenizer=AutoTokenizer.from_pretrained("Rostlab/prot_bert"),
    return_tensors=True,
    device=0 if torch.cuda.is_available() else -1,
)
smile_model = MyFeatureExtractionPipeline(
    task="feature-extraction",
    model=AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k"),
    tokenizer=AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k"),
    return_tensors=True,
    device=0 if torch.cuda.is_available() else -1,
)


max_inp1_len = 512
max_inp2_len = 64


def custom_format(data):
    [inp1, inp2, inp3], label = data
    if inp1.shape[1] >= max_inp1_len:
        inp1 = F.adaptive_avg_pool2d(inp1, (max_inp1_len, 1024))
    if inp2.shape[1] >= max_inp2_len:
        inp2 = F.adaptive_avg_pool2d(inp2, (max_inp2_len, 768))
        inp3 = F.adaptive_avg_pool2d(inp3, (max_inp2_len, 1024))

    return [inp1, inp2, inp3], label


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        patch_dim,
        num_patches,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="mean",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        return x


class DTI_MOA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self._dim = 512
        self.linear1 = nn.Linear(1024, self._dim)
        self.linear2 = nn.Linear(768, self._dim)
        self.attn1 = nn.MultiheadAttention(self._dim, 8, batch_first=True)
        self.attn2 = nn.MultiheadAttention(self._dim, 8, batch_first=True)
        self.linear3 = nn.Linear(self._dim, self._dim)
        self.vit = ViT(self._dim, 2 * max_inp2_len, self._dim, 4, 8, self._dim)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.linear4 = nn.Linear(self._dim, self._dim)
        self.linear5 = nn.Linear(self._dim, num_classes)

    def forward(self, x1, x2, x3):
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x3 = self.linear1(x3)
        x2 = x2 + x3
        x3 = self.attn1(x1, x2, x2)[0]
        x2 = self.attn2(x2, x1, x1)[0]
        x1 = F.adaptive_avg_pool2d(x3, (max_inp2_len, self._dim))
        x1 = self.linear3(x1)
        x = torch.concat([x1, x2], dim=1)
        x = self.vit(x)
        x = self.dropout(x)
        x = F.relu(self.linear4(x))
        x = self.linear5(x)
        return x


model = DTI_MOA(7)
model.load_state_dict(torch.load("model_state_dict.pt", device))
model.eval()

class_map = {    0: "inhibitor",    1: "antagonist",    2: "agonist",    3: "ligand",    4: "allosteric modulator",    5: "inducer",    6: "unknown",}


def get_dti_moa(uniprot_id, db_id):
    try:
        row = df.loc[df["uniprotkb-id"] == uniprot_id].iloc[0]
    except IndexError:
        return "UniProt-ID not found in database"
    prot = row["aa-seq"]
    prot = " ".join(list(prot))
    prot = prot_model(prot)

    try:
        row = df.loc[df["drugbank-id"] == db_id].iloc[0]
    except IndexError:
        return "DrugBank-ID not found in database"
    if row["type"] == "Protein":
        prot_drug = row["structure"]
        prot_drug = " ".join(list(prot_drug))
        prot_drug = prot_model(prot_drug)
        chem_drug = torch.zeros((*prot_drug.shape[:-1], 768))
    else:
        chem_drug = row["structure"]
        chem_drug = smile_model(chem_drug)
        prot_drug = torch.zeros((*chem_drug.shape[:-1], prot.shape[-1]))

    with torch.no_grad():
        out = model(*(custom_format(([prot, chem_drug, prot_drug], None))[0]))

    out = torch.softmax(out, -1).flatten().detach().cpu().numpy()
    out_dict = {}
    for i, p in enumerate(out):
        out_dict[class_map[i]] = p
    return out_dict

db_id = input('Enter Drugbank ID: ')
uniprot_id = input('Enter potential target UniProtKB ID: ')
input('Press Enter to continue...')

res = get_dti_moa(uniprot_id, db_id)
print("\n")
for k,v in res.items():
    print(k + ": \t" + str(round(v*100, 2)) + "%")