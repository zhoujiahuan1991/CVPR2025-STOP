import torch
import torch.nn as nn

def get_TemporalPrompt(args):
    if args.temporal_prompt in ['group2-2']:
        return TemporalPrompt_3(args=args)
   

class TemporalPrompt_3(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        kernel_spatial = 11
        kernel_temporal = 3
        kernel = (kernel_temporal, kernel_spatial, kernel_spatial)
        padding = (int((kernel_temporal-1)/2), int((kernel_spatial-1)/2), int((kernel_spatial-1)/2))
        hid_dim_1 = 9
        hid_dim_2 = 9
        hid_dim_3 = 16
        hid_dim_l1 = 16
        
        self.Conv = nn.Sequential(
            nn.Conv3d(3, hid_dim_1, kernel_size=kernel, stride=1, padding=padding),
            nn.Conv3d(hid_dim_1, hid_dim_1, kernel_size=kernel, stride=1, padding=padding),
            nn.PReLU(),
            nn.Conv3d(hid_dim_1, hid_dim_2, kernel_size=kernel, stride=1, padding=padding),
            nn.PReLU(),
            nn.Conv3d(hid_dim_2, hid_dim_3, kernel_size=kernel, stride=1, padding=padding),
        )
        self.MLP = nn.Sequential(
            nn.Linear(hid_dim_3, hid_dim_l1),
            nn.PReLU(),
            nn.Linear(hid_dim_l1, 3),
        )
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        
        padding_tmp = (int((5-1)/2), int((11-1)/2), int((11-1)/2))
        self.Cal_Net = nn.Conv3d(3, 1, kernel_size=(5, 11, 11), stride=1, padding=padding_tmp)
        self.eta = 6
        
        self.InterFramePrompt = self.init_InterFramePrompt(args)
        
    def forward(self, x):
        B, T, C, W, H = x.shape
        prompt = x.permute(0, 2, 1, 3, 4)
        prompt = self.Conv(prompt)
        prompt = self.dropout1(prompt)
        prompt = prompt.permute(0, 2, 3, 4, 1)
        prompt = self.MLP(prompt)
        prompt = self.dropout2(prompt)
        prompt = prompt.permute(0, 1, 4, 2, 3)
        prompt = self.dropout2(prompt)
        mask = self.get_mask(x)
        return x + prompt*0.05 + prompt*mask*0.05
    
    def get_mask(self, x):
        B, T, C, W, H = x.shape
        self.B, self.T = B, T
        x = x.permute(0, 2, 1, 3, 4)
        x = self.Cal_Net(x)
        x = x.squeeze(1)
        x = x.reshape(B, T, 32, 7, 32, 7)
        x = x.mean(dim=(2, 4))
        bar = x.reshape(B, T, -1)
        bar = bar.sort(dim=2, descending=True)[0]
        bar = bar[:, :, self.eta]
        x = x > bar.unsqueeze(2).unsqueeze(3)
        self.mask = x
        x = x.unsqueeze(2).unsqueeze(4)
        x = x.repeat(1, 1, 32, 1, 32, 1)
        x = x.reshape(B, T, 224, 224)
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, 3, 1, 1)
        return x
    
    def init_InterFramePrompt(self, args):
        self.Attention = nn.MultiheadAttention(embed_dim = 768, num_heads = 12)
        kernel_temporal = 3 
        kernel_token = 15
        kernel_hid = 25
        kernel = (kernel_temporal, kernel_token, kernel_hid)
        padding = (int((kernel_temporal-1)/2), int((kernel_token-1)/2), int((kernel_hid-1)/2))
        hid_dim_1 = 9
        hid_dim_l1 = 16
        
        self.InterConv = nn.Sequential(
            nn.Conv3d(1, hid_dim_1, kernel_size=kernel, stride=1, padding=padding),
            nn.PReLU(),
            nn.Conv3d(hid_dim_1, 1, kernel_size=kernel, stride=1, padding=padding),
        )
        self.InterMLP = nn.Sequential(
            nn.Linear(49, 16),
            nn.PReLU(),
            nn.Linear(16, 4),
        )
        padding_tmp = (int((5-1)/2), int((11-1)/2), int((11-1)/2))
        self.Cal_Net_Inter = nn.Conv3d(1, 1, kernel_size=(5, 11, 11), stride=1, padding=padding_tmp)
        
    def get_mask_Inter(self, x):
        x = self.Cal_Net_Inter(x)
        x = x.squeeze(1)
        mask_tmp = self.mask.reshape(self.B, self.T, -1)
        mask_tmp = mask_tmp.unsqueeze(3)
        x = x + x*mask_tmp
        x = x.mean(dim=(2, 3))
        x = (x-x.min())/(x.max()-x.min())
        return x

    
    def get_inter_frame_prompt(self, x):
        BT, L, D = x.shape
        x = x.reshape(self.B, self.T, L, D)
        x = x[:,:,1:,:]
        x = x.reshape(self.B, -1, D)
        x = x.permute(1, 0, 2)
        x = self.Attention(x, x, x)[0]
        x = x.permute(1, 0, 2)
        x = x.reshape(self.B, self.T, -1, D)
        x = x.unsqueeze(1)
        mask = self.get_mask_Inter(x)
        x = self.InterConv(x)
        x = x.squeeze(1)
        x = x.permute(0, 1, 3, 2)
        x = self.InterMLP(x)
        x = x.permute(0, 1, 3, 2)
        mask = mask.unsqueeze(2).unsqueeze(3)
        x = x+x*mask
        x = x.reshape(self.B, -1, 768)
        x = x.unsqueeze(0)
        x = x.repeat(12, 1, 1, 1)
        return x*0.05
    
    
    
