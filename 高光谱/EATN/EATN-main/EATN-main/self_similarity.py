import torch
import torch.nn as nn
import torch.nn.functional as F


class SSFE(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, unfold_size=3, ksize=3):
        super(SSFE, self).__init__()

        self.SCC = SelfCorrelationComputation(unfold_size=unfold_size)
        self.encoder_q = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False, padding=0)
        self.SSE = SelfSimilarityEncoder(mid_ch, unfold_size=unfold_size, ksize=ksize)
        self.FFN = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True, padding=0),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True, padding=0))

    def forward(self, ssm_input_feat):
        batch_size, c, h, w = ssm_input_feat.size()
        q = self.encoder_q(ssm_input_feat)
        q = F.normalize(q, dim=1, p=2)
        q1, k = q.view(batch_size, -1, h * w).permute(0, 2, 1), q.view(batch_size, -1, h * w)
        cent_spec_vector = q1[:, int((h * w - 1) / 2)]
        cent_spec_vector = torch.unsqueeze(cent_spec_vector, 1)
        sim_cos = F.cosine_similarity(cent_spec_vector, k.permute(0, 2, 1), dim=2)  # include negative
        sim_cos = sim_cos.clamp(min=0)
        atten_s = torch.unsqueeze(sim_cos, 2)
        q_attened = torch.mul(atten_s, q1)
        out = q_attened.contiguous().view(batch_size, -1, h, w)
        self_sim = self.SCC(q)
        self_sim_feat = self.SSE(self_sim)
        ssm_output_feat = ssm_input_feat + self_sim_feat + out
        ssm_output_feat = self.FFN(ssm_output_feat)

        return ssm_output_feat


class SelfCorrelationComputation(nn.Module):
    def __init__(self, unfold_size=5):
        super(SelfCorrelationComputation, self).__init__()
        self.unfold_size = (unfold_size, unfold_size)
        self.padding_size = unfold_size // 2
        self.unfold = nn.Unfold(kernel_size=self.unfold_size, padding=self.padding_size)

    def forward(self, q):
        b, c, h, w = q.shape
        q_unfold = self.unfold(q)  # b, cuv, h, w
        q_unfold = q_unfold.view(b, c, self.unfold_size[0], self.unfold_size[1], h, w) # b, c, u, v, h, w
        self_sim = q_unfold * q.unsqueeze(2).unsqueeze(2)  # b, c, u, v, h, w * b, c, 1, 1, h, w
        self_sim = self_sim.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v
        return self_sim.clamp(min=0)
        
class SelfSimilarityEncoder(nn.Module):
    def __init__(self, mid_ch, unfold_size, ksize):
        super(SelfSimilarityEncoder, self).__init__()
            
        def make_building_conv_block(in_channel, out_channel, ksize, padding=(0,0,0), stride=(1,1,1), bias=True, conv_group=1):
            building_block_layers = []
            building_block_layers.append(nn.Conv3d(in_channel, out_channel, (1, ksize, ksize),
                                             stride=stride, bias=bias, groups=conv_group, padding=padding))
            building_block_layers.append(nn.BatchNorm3d(out_channel))
            building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        conv_in_block_list = [make_building_conv_block(mid_ch, mid_ch, ksize) for _ in range(unfold_size//2)]
        self.conv_in = nn.Sequential(*conv_in_block_list)


    def forward(self, x):
        b, c, h, w, u, v = x.shape

        x = x.view(b, c, h * w, u, v)
        x = self.conv_in(x)
        c = x.shape[1]
        x = x.mean(dim=[-1,-2]).view(b, c, h, w)
        return x



