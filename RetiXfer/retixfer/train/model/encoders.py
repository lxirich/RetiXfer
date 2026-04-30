import torch
import torchvision
import math
from torch import nn
from transformers import AutoModel, AutoTokenizer 
from .ViT import VisionTransformer
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VisionModel(nn.Module):
    def __init__(self, vision_type='resnet', patch_size = 32, pretrained=True, proj_dim=512, 
                 proj_bias=False, projection=True,norm=True):
        super().__init__()
        self.proj_dim = proj_dim
        self.patch_size = patch_size
        self.vision_type = vision_type

        if self.vision_type not in ['resnet_v1', 'resnet_v2', 'efficientnet', 'ViT_B_512', 'ViT_S_512']:
            print(f"[Warning] Unknown vision_type {self.vision_type}, using 'resnet_v1'.")
            self.vision_type = "resnet_v1"

        if self.vision_type in ["resnet_v1", "resnet_v2"]:
            weights = 'IMAGENET1K_V1' if self.vision_type=="resnet_v1" else 'IMAGENET1K_V2'
            weights = weights if pretrained else None
            self.model = torchvision.models.resnet50(weights=weights)
            self.model.fc = nn.Identity()
            self.vision_dim = 2048
        elif self.vision_type == "efficientnet":
            weights = 'IMAGENET1K_V1' if pretrained else None
            self.model = torchvision.models.efficientnet_b7(weights=weights)
            self.vision_dim = 2096
        elif self.vision_type == "ViT_B_512":
            self.model = VisionTransformer(patch_size=self.patch_size, width=768, layers=12, heads=12)
            self.vision_dim = self.model.num_features
        elif self.vision_type == "ViT_S_512":
            self.model = VisionTransformer(patch_size=self.patch_size, width=384, layers=12, heads=6)
            self.vision_dim = self.model.num_features

        if projection:
            self.projection_head = ProjectionLayer(
                layer=nn.Linear(self.vision_dim, self.proj_dim, bias=proj_bias),
                projection=projection, norm=norm
            )

    def forward(self, x):
        if self.vision_type.startswith("ViT"):
            embed = self.model(x)
            return self.projection_head(embed)
        else:
            embed = self.model(x)
            return self.projection_head(embed)


class TextModel(torch.nn.Module):
    def __init__(self, bert_type='emilyalsentzer/Bio_ClinicalBERT', proj_dim=512, proj_bias=False, 
                 projection=True,norm=True, text_pretrained=True, text_weights_path=None):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.tokenizer.model_max_length = 256
        self.model = AutoModel.from_pretrained(bert_type, output_hidden_states=True)

        if text_pretrained and text_weights_path:
            self.load_from_pretrained(text_weights_path)

        self.projection_head = ProjectionLayer(
            layer=nn.Linear(768, proj_dim, bias=proj_bias),
            projection=projection, norm=norm
        )
        
    def load_from_pretrained(self, weights_path):
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        new_state_dict = {k.replace('bert', 'model'): v for k, v in state_dict.items() if 'cls' not in k}
        self.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded text weights from {weights_path}")

    def tokenize(self, prompts_list):
        text_tokens = self.tokenizer(prompts_list, truncation=True, padding=True, return_tensors='pt')
        return text_tokens

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = torch.stack([outputs['hidden_states'][1],
                                   outputs['hidden_states'][2],
                                   outputs['hidden_states'][-1]])
        embed = last_hidden.permute(1,0,2,3).mean(2).mean(1)
        embed = self.projection_head(embed)
        return embed


class ProjectionLayer(nn.Module):
    def __init__(self, layer, projection=True, norm=True):
        super().__init__()
        self.apply_projection = projection
        self.norm_modality = projection and norm
        self.norm_projection = norm
        self.projection = layer

    def forward(self, x):
        if self.norm_modality:
            x = F.normalize(x, p=2, dim=-1)
        if self.apply_projection:
            x = self.projection(x)
            if self.norm_projection:
                x = F.normalize(x, p=2, dim=-1)
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.W_q = torch.nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = torch.nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = torch.nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = torch.nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.dropout = torch.nn.Dropout(dropout)

    def transpose_qkv(self, X, num_heads):
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def forward(self, queries, keys, values):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads) 
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)   
        values = self.transpose_qkv(self.W_v(values), self.num_heads)  
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = torch.nn.functional.softmax(scores)       
        output = torch.bmm(self.dropout(self.attention_weights), values)
        output_concat = self.transpose_output(output, self.num_heads)  

        return self.W_o(output_concat)


class PositionWiseFFN(torch.nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = torch.nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(torch.nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.ln = torch.nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

