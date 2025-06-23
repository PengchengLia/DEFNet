from functools import partial
from turtle import forward
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F



from functools import partial
from turtle import forward
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F

###output_newer_again

class Fusion(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.fun = Fun()

    def forward(self, x_vis, x_inf):
        f = self.fun(x_vis, x_inf)
        return f
    
class Fun(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        head_dim = dim // num_heads
        self.norm = norm_layer(dim)
        # self.norm2 = norm_layer(head_dim)
        mlp_hidden_dim = int(head_dim * mlp_ratio)
        self.mlp = Mlp(in_features=head_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.scale = head_dim ** -0.5
        self.D1 = Difference_Module() 
        self.D2 = Difference_Module() 
        self.v_1 = None
        self.v_2 = None
        
        

    def forward(self, x_vis, x_inf):
        # x_v: [B, N, C], N = 320
        # x_i: [B, N, C], N = 320
        B, N, C = x_vis.shape
        qkv_vis = self.qkv(self.norm(x_vis)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_inf = self.qkv(self.norm(x_inf)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_vis, k_vis, v_vis = qkv_vis.unbind(0)   #24 12 320 64
        q_inf, k_inf, v_inf = qkv_inf.unbind(0)
        v_1 = self.D1(q_vis,k_inf,v_inf)
        v_2 = self.D2(q_inf,k_vis,v_vis)
        v_fuse = v_1 + v_2
        self.v_1 = v_1
        self.v_2 = v_2
        return v_fuse.transpose(1, 2).reshape(B, N, C)


class Difference_Module(nn.Module):
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
        self.v1 = None

    def forward(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = attn @ v
        self.v1 = attn
        v1 = self.dif_proj(v - attn)
        v_new = ((q @ k.transpose(-2, -1)) * self.scale) @ v1
        v_new = v_new + q
        v_new = self.mlp(self.norm(v_new)) + v_new
        return v_new
    


class Fun1(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.DIIM1 = Difference_Module1()
        
    def forward(self, x_vis, x_inf):
        attn1, attn2 = self.DIIM1(x_vis, x_inf)

        return attn1, attn2

class Difference_Module1(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.qkv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)

        self.dif_proj1 = nn.Linear(dim//num_heads,dim//num_heads)
        self.dif_proj2 = nn.Linear(dim//num_heads,dim//num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm = norm_layer(dim)

        self.attn_before_v = None
        self.attn_before_i = None
        self.attn_after_v = None
        self.attn_after_i = None
        self.attn_fuse = None


    def forward(self, x_vis, x_inf):
        B, N, C = x_vis.shape
        qkv_vis = self.qkv1(self.norm(x_vis)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_inf = self.qkv2(self.norm(x_inf)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_vis, k_vis, v_vis = qkv_vis.unbind(0)   #24 12 320 64
        q_inf, k_inf, v_inf = qkv_inf.unbind(0)


        attn_vis = (q_vis @ (k_vis.transpose(-2, -1))) * self.scale
        # attn_vis = attn_vis.softmax(dim=-1)

        attn_inf = (q_inf @ (k_inf.transpose(-2, -1))) * self.scale
        # attn_inf = attn_inf.softmax(dim=-1)

        self.attn_before_v = attn_vis.softmax(dim=-1)
        self.attn_before_i = attn_inf.softmax(dim=-1)

        attn_fuse = attn_vis * attn_inf
        # attn_fuse = attn_fuse.softmax(dim=-1)
        # attn_fuse = self.attn_drop(attn_fuse)
        self.attn_fuse = attn_fuse.softmax(dim=-1)

        attn_dif_vis = attn_vis - attn_fuse
        attn_dif_inf = attn_inf - attn_fuse

        self.attn_after_v = attn_dif_vis.softmax(dim=-1)
        self.attn_after_i = attn_dif_inf.softmax(dim=-1)

        attn_dif_vis = attn_dif_vis.softmax(dim=-1)
        attn_dif_vis = self.attn_drop(attn_dif_vis)
        attn_dif_vis = attn_dif_vis @ v_vis
        attn_dif_vis = self.dif_proj1(attn_dif_vis)
        attn_dif_vis = attn_dif_vis.transpose(1, 2).reshape(B, N, C) + (x_vis)
        attn_dif_vis = self.mlp1(self.norm(attn_dif_vis)) + attn_dif_vis


        attn_dif_inf = attn_dif_inf.softmax(dim=-1)
        attn_dif_inf = self.attn_drop(attn_dif_inf)
        attn_dif_inf = attn_dif_inf @ v_inf
        attn_dif_inf = self.dif_proj2(attn_dif_inf)
        attn_dif_inf = attn_dif_inf.transpose(1, 2).reshape(B, N, C) + (x_inf)
        attn_dif_inf = self.mlp2(self.norm(attn_dif_inf)) + attn_dif_inf  
        
        return attn_dif_vis, attn_dif_inf
