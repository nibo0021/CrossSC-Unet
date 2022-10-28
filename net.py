import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(w, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output
    
    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0], relation[:,:,1]]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x

class Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self,in_ch=3, num_classes=1, config=[2,2,18,2], dim=128, drop_path_rate=0.2, input_resolution=192):
        super().__init__()
        # n1 = 64
        # filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        filters = [64, 128, 256, 512, 1024]
        
        self.Maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv0 = conv_block(in_ch, 32)
        self.Conv1 = conv_block(32, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.merge0 = conv_block(filters[1] * 2, filters[1])
        self.merge1 = conv_block(filters[2] * 2, filters[2])
        self.merge2 = conv_block(filters[3] * 2, filters[3])
        self.merge3 = conv_block(filters[4] * 2, filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[0] * 2, filters[0])

        self.Up1 = up_conv(filters[0], 32)
        self.Up_conv1 = conv_block(32, 3)

        self.out = nn.Conv2d(3, num_classes, kernel_size=1, stride=1, padding=0)


        self.config = config
        self.dim = dim
        self.head_dim = 32
        self.window_size = 6
        # self.patch_partition = Rearrange('b c (h1 sub_h) (w1 sub_w) -> b h1 w1 (c sub_h sub_w)', sub_h=4, sub_w=4)

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        begin = 0
        self.stage1 = [nn.Conv2d(3, dim, kernel_size=4, stride=4),
                       Rearrange('b c h w -> b h w c'),
                       nn.LayerNorm(dim),] + \
                      [Block(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//4) 
                      for i in range(config[0])] + \
                      [Rearrange('b h w c -> b c h w')]
        begin += config[0]
        self.stage2 = [Rearrange('b c (h neih) (w neiw) -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(4*dim), nn.Linear(4*dim, 2*dim, bias=False),] + \
                      [Block(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                      for i in range(config[1])] + \
                      [Rearrange('b h w c -> b c h w')]
        begin += config[1]
        self.stage3 = [Rearrange('b c (h neih) (w neiw) -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(8*dim), nn.Linear(8*dim, 4*dim, bias=False),] + \
                      [Block(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//16)
                      for i in range(config[2])] + \
                      [Rearrange('b h w c -> b c h w')]
                      
        begin += config[2]
        self.stage4 = [Rearrange('b c (h neih) (w neiw) -> b h w (neiw neih c)', neih=2, neiw=2), 
                       nn.LayerNorm(16*dim), nn.Linear(16*dim, 8*dim, bias=False),] + \
                      [Block(8*dim, 8*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//32)
                      for i in range(config[3])] + \
                      [Rearrange('b h w c -> b c h w')]
        
        self.stage1 = nn.Sequential(*self.stage1)
        self.stage2 = nn.Sequential(*self.stage2)
        self.stage3 = nn.Sequential(*self.stage3)
        self.stage4 = nn.Sequential(*self.stage4)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        
        x1 = self.Conv0(x)
        x1 = self.Maxpool0(x1)
        
        e1 = self.Conv1(x1)
        e2 = self.Maxpool1(e1)
        
        e2 = self.Conv2(e2)
        s1 = self.stage1(x)
        m0 = self.merge0(torch.cat((e2, s1), dim = 1)) 
        # print(e2.shape, s1.shape, m0.shape)

        s1 = self.Maxpool2(s1)
        e3 = self.Conv3(s1)
        s2 = self.stage2(e2)
        m1 = self.merge1(torch.cat((e3, s2), dim = 1))
        # print(e3.shape, s2.shape, m1.shape)


        s2 = self.Maxpool3(s2)
        e4 = self.Conv4(s2)
        s3 = self.stage3(e3)
        m2 = self.merge2(torch.cat((e4, s3), dim = 1))
        # print(e4.shape, s3.shape, m2.shape)


        s3 = self.Maxpool4(s3)
        e5 = self.Conv5(s3)
        s4 = self.stage4(e4)
        m3 = self.merge3(torch.cat((e5, s4), dim = 1))
        # print(e5.shape, s4.shape, m3.shape)


        d5 = self.Up5(m3)
        d5 = torch.cat((m2, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((m1, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((m0, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = self.Up_conv1(d1)

        return self.out(d1)
