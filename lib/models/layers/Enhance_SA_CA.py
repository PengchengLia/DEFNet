from functools import partial
from turtle import forward
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.fun = Fun()
        # self.fusion2 = Fusion()

    def forward(self, x_vis, x_inf):
        f = self.fun(x_vis, x_inf)
        return f
    
class Fun(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        self.norm = norm_layer(dim)
        # self.scale = head_dim ** -0.5
        self.NF1 = New_Feature()
        self.NF2 = New_Feature()
        self.Es_SA1 = Enhance_SA()
        self.Es_SA2 = Enhance_SA()
        self.Es_CA1 = Enhance_CA()
        self.Es_CA2 = Enhance_CA()
        

    def forward(self, x_vis, x_inf):
        # x_v: [B, N, C], N = 320
        # x_i: [B, N, C], N = 320
        B, N, C = x_vis.shape
        qkv_vis = self.qkv(self.norm(x_vis)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_inf = self.qkv(self.norm(x_inf)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_vis, k_vis, v_vis = qkv_vis.unbind(0)   #24 12 320 64
        q_inf, k_inf, v_inf = qkv_inf.unbind(0)
        
        ##增强自注意力部分
        Process_Feature1 = self.NF1(q_vis,k_inf,v_inf)
        Enhance_inf_SA = self.Es_SA1(q_inf,k_inf,Process_Feature1)
        Process_Feature2 = self.NF2(q_inf,k_vis,v_vis)
        Enhance_vis_SA = self.Es_SA2(q_vis,k_vis,Process_Feature2)

        ##增强交叉注意力部分
        Enhance_vis_CA = self.Es_CA1(q_vis,k_inf,Process_Feature1)
        Enhance_inf_CA = self.Es_CA2(q_inf,k_vis,Process_Feature2)

        ##分别融合各自模态的增强自注意力和增强交叉注意力
        Enhance_fus1 = Enhance_inf_CA + Enhance_inf_SA
        Enhance_fus2 = Enhance_vis_CA + Enhance_vis_SA

        return Enhance_fus1.transpose(1, 2).reshape(B, N, C) + Enhance_fus2.transpose(1, 2).reshape(B, N, C)

class New_Feature(nn.Module):
    def __init__(self, dim=64, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

        self.dif_proj = nn.Linear(dim,dim)
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn) 
        attn = attn @ v #原始交叉注意力
        v1 = self.dif_proj(v - attn) #一个消除了交互特征以后的新的独有特征
        return v1


class Enhance_SA(nn.Module):
    def __init__(self, dim=64, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

        self.dif_proj = nn.Linear(dim,dim)
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    ###这里增强自注意力
    def forward(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn) 
        attn = attn @ v #原始交叉注意力
        v1 = self.dif_proj(v - attn) #一个消除了交互特征以后的新的独有特征
        sa_new = ((q @ k.transpose(-2, -1)) * self.scale) @ v1 
        sa_new = self.mlp(self.norm(sa_new)) + sa_new
        return sa_new
    

class Enhance_CA(nn.Module):
    def __init__(self, dim=64, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

        self.dif_proj = nn.Linear(dim,dim)
        self.norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    ###这里增强交叉注意力
    def forward(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) 
        attn = self.attn_drop(attn) 
        attn = attn @ v #原始交叉注意力
        v1 = self.dif_proj(v - attn) #一个消除了交互特征以后的新的独有特征
        ca_new = ((q @ k.transpose(-2, -1)) * self.scale) @ v1 
        ca_new = ca_new + q
        ca_new = self.mlp(self.norm(ca_new)) + ca_new
        return ca_new



# if __name__ == '__main__':
#     q = torch.rand(24,12,320,64)
#     k = torch.rand(24,12,320,64)
#     v = torch.rand(24,12,320,64)
#     F1 = Difference_Module()
#     f = F1(q,k,v)