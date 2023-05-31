import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.SubLayers import PositionwiseFeedForward, AddNorm


class CoTransformerEncoder(nn.Module):
    '''Co-Attention Transformer encoder. Inspired by ViLBERT model'''

    def __init__(self, q_dim, v_dim, num_heads = 12, hidden_dim=2048, drop_out=0.1) -> None:
        super(CoTransformerEncoder, self).__init__()
        # Transformer Encoder 1 for Visual features
        self.mhsa_1 = nn.MultiheadAttention(v_dim, num_heads, kdim=q_dim, vdim=q_dim, batch_first=True, dropout=drop_out)
        self.ffw_1 = PositionwiseFeedForward(v_dim, hidden_dim)
        self.addnorm_1_1 = AddNorm(v_dim, drop_out)
        self.addnorm_1_2 = AddNorm(v_dim, drop_out)
        
        # Transformer Encoder 2 for Question features
        self.mhsa_2 = nn.MultiheadAttention(q_dim, num_heads, kdim=v_dim, vdim=v_dim, batch_first=True, dropout=drop_out)
        self.ffw_2 = PositionwiseFeedForward(q_dim, hidden_dim)
        self.addnorm_2_1 = AddNorm(q_dim, drop_out)
        self.addnorm_2_2 = AddNorm(q_dim, drop_out)
            
    def forward(self, v, q):
        ''' Forward
        v: [batch, k, v_dim]
        q: [batch, seq_len, q_dim]
        '''
        x_1, _ = self.mhsa_1(v, q, q)
        x_1 = self.addnorm_1_1(v, x_1)
        out_1 = self.addnorm_1_2(x_1, self.ffw_1(x_1))
        
        x_2, _ = self.mhsa_2(q, v, v)
        x_2 = self.addnorm_2_1(q, x_2)
        out_2 = self.addnorm_2_2(x_2, self.ffw_2(x_2))
        
        return out_1, out_2


class CoTransformerBlock(nn.Module):
    def __init__(self, v_dim, q_dim, num_head_trfm=12, hidden_size=2048, drop_out=0.1) -> None:
        super(CoTransformerBlock, self).__init__()
        self.co_trm_encoder = CoTransformerEncoder(q_dim, v_dim, num_head_trfm, hidden_size, drop_out)
        self.transformer_1 = nn.TransformerEncoderLayer(v_dim, nhead=num_head_trfm, dim_feedforward=hidden_size, dropout=drop_out, batch_first=True)
        self.transformer_2 = nn.TransformerEncoderLayer(q_dim, nhead=num_head_trfm, dim_feedforward=hidden_size, dropout=drop_out, batch_first=True)
        
    def forward(self, v, q):
        ''' Forward
        v: [batch, k, v_dim]
        q: [batch, seq_len, q_dim]
        '''
        v_out, q_out = self.co_trm_encoder(v, q)
        v_out = self.transformer_1(v_out)
        q_out = self.transformer_2(q_out)
        return v_out, q_out


class GuidedTransformerEncoder(nn.Module):
    '''Guided-Attention Transformer encoder'''

    def __init__(self, q_dim, kv_dim, num_heads=12, hidden_dim=2048, drop_out=0.1) -> None:
        super(GuidedTransformerEncoder, self).__init__()
        self.mhsa_1 = nn.MultiheadAttention(q_dim, num_heads, kdim=kv_dim, vdim=kv_dim, batch_first=True, dropout=drop_out)
        self.ffw_1 = PositionwiseFeedForward(q_dim, hidden_dim)
        self.addnorm_1_1 = AddNorm(q_dim, drop_out)
        self.addnorm_1_2 = AddNorm(q_dim, drop_out)

            
    def forward(self, q, kv):
        ''' Forward
        q: query: [batch, q_seq_length, q_dim]
        kv: key & value: [batch, q_kv_length, kv_dim]
        return: [batch, q_seq_length, q_dim]
        '''
        x_1, _ = self.mhsa_1(q, kv, kv)
        x_1 = self.addnorm_1_1(q, x_1)
        out_1 = self.addnorm_1_2(x_1, self.ffw_1(x_1))

        return out_1


class FusionAttentionFeature(nn.Module):
    def __init__(self, args) -> None:
        super(FusionAttentionFeature, self).__init__()
        
        self.v_convert = nn.Sequential(*[
            nn.Linear(args.v_dim, args.f_mid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.f_mid_dim, args.joint_dim)
        ])
                
        self.q_convert = nn.Sequential(*[
            nn.Linear(args.q_dim, args.f_mid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.f_mid_dim, args.joint_dim)
        ])
        
        self.layer_norm = nn.LayerNorm(args.joint_dim)

    def forward(self, v_feat, q_feat):
        '''Forward
        v_feat: [batch, v_len, v_dim]
        q_feat: [batch, q_len, q_dim]
        '''
        v_converted = self.v_convert(v_feat)
        q_converted = self.q_convert(q_feat)

        out = self.layer_norm(v_converted + q_converted)

        return out


class AttentionReduce(nn.Module):
    def __init__(self, dim, mid_dim, glimpses):
        super(AttentionReduce, self).__init__()
        self.mlp = nn.Sequential(*[
            nn.Linear(dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_dim, glimpses)
        ])
        self.glimpses = glimpses
    
    def forward(self, x, y):
        x_reduced = self.mlp(x)
        x_reduced = F.softmax(x_reduced, dim=1)
        
        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(x_reduced[:, :, i: i + 1] * y, dim=1)
            )
        x_atted = torch.cat(att_list, dim=1)
        
        return x_atted


class FusionAttentionReduce(nn.Module):
    def __init__(self, args) -> None:
        super(FusionAttentionReduce, self).__init__()
        self.q_convert = nn.Sequential(*[
            nn.Linear(args.q_dim, args.f_mid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.f_mid_dim, args.joint_dim)
        ])
        self.v_convert = nn.Sequential(*[
            nn.Linear(args.v_dim, args.f_mid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.f_mid_dim, args.joint_dim)
        ])
        self.layer_norm = nn.LayerNorm(args.joint_dim)

    def forward(self, v_feat, q_feat):
        '''Forward
        v_feat: [batch, v_len, v_dim]
        q_feat: [batch, q_len, q_dim]
        '''
        v_converted = self.v_convert(v_feat)
        v_att = F.softmax(v_converted, dim=1)
        v_atted = torch.sum(v_att * v_feat, dim=1)

        q_converted = self.q_convert(q_feat)
        q_att = F.softmax(q_converted, dim=1)
        q_atted = torch.sum(q_att * q_feat, dim=1)
        
        out = self.layer_norm(v_atted * q_atted)

        return out
        
class FusionLinear(nn.Module):
    def __init__(self, in_dim_1, in_dim_2, out_dim) -> None:
        super(FusionLinear, self).__init__()
        self.ffn_1 = nn.Linear(in_dim_1, out_dim)
        self.ffn_2 = nn.Linear(in_dim_2, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x1, x2):
        '''Forward
        v_feat: [batch, v_len, v_dim]
        q_feat: [batch, q_len, q_dim]
        '''
        return self.layer_norm(self.ffn_1(x1) * self.ffn_2(x2))
        