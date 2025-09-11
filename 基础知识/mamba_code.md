# DualMamba
```python
class SpeMamba(nn.Module):  
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):  
        super(SpeMamba, self).__init__()  
        self.token_num = token_num  
        self.use_residual = use_residual  
  
        self.group_channel_num = math.ceil(channels / token_num)  
        self.channel_num = self.token_num * self.group_channel_num  
  
        self.mamba = Mamba(  
            d_model=self.group_channel_num,  
            d_state=16,  
            d_conv=4,  
            expand=2,  
        )  
  
        self.proj = nn.Sequential(  
            nn.GroupNorm(group_num, self.channel_num),  
            nn.SiLU()  
        )  
  
    def padding_feature(self, x):  
        B, C, H, W = x.shape  
        if C < self.channel_num:  
            pad_c = self.channel_num - C  
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)  
            cat_features = torch.cat([x, pad_features], dim=1)  
            return cat_features.contiguous()  # 确保填充后连续  
        else:  
            return x.contiguous()  
  
    def forward(self, x):  
        x_pad = self.padding_feature(x)  
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()  # 确保permute后连续  
        B, H, W, C_pad = x_pad.shape  
        x_flat = x_pad.view(B * H * W, self.token_num, self.group_channel_num).contiguous()  # 确保连续  
        x_flat = self.mamba(x_flat)  
        x_recon = x_flat.view(B, H, W, C_pad).contiguous()  # 确保view后连续  
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # 确保permute后连续  
        x_proj = self.proj(x_recon).contiguous()  # 确保投影后连续  
  
        if self.use_residual:  
            return (x + x_proj).contiguous()  # 确保残差连接后连续  
        else:  
            return x_proj.contiguous()
            
class SpaMamba(nn.Module):  
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True):  
        super(SpaMamba, self).__init__()  
        self.use_residual = use_residual  
        self.use_proj = use_proj  
        self.mamba = Mamba(  
            d_model=channels,  
            d_state=16,  
            d_conv=4,  
            expand=2,  
        )  
        if self.use_proj:  
            self.proj = nn.Sequential(  
                nn.GroupNorm(group_num, channels),  
                nn.SiLU()  
            )  
  
    def forward(self, x):  
        x_re = x.permute(0, 2, 3, 1).contiguous()  # 确保permute后连续  
        B, H, W, C = x_re.shape  
        x_flat = x_re.view(1, -1, C).contiguous()  # 确保view后连续  
        x_flat = self.mamba(x_flat)  
        x_recon = x_flat.view(B, H, W, C).contiguous()  # 确保view后连续  
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()  # 确保permute后连续  
  
        if self.use_proj:  
            x_recon = self.proj(x_recon).contiguous()  # 确保投影后连续  
  
        if self.use_residual:  
            return (x_recon + x).contiguous()  # 确保残差连接后连续  
        else:  
            return x_recon.contiguous()
```