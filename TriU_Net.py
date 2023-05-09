import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchaudio
from sConformer import sConformer


class ConvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(2,3), stride=(1,2)) -> None:
        super(ConvGLU, self).__init__()

        self.main = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

        self.pad = nn.ZeroPad2d((0,0,1,0)) # pad 0 on the start of Time axis
    
    def forward(self, x):
        x = self.pad(x)
        main = self.main(x)
        gate = self.gate(x)

        return main * torch.sigmoid(gate)


class DeconvGLU(nn.Module):
    def __init__(self, in_ch, out_ch, pad, kernel_size=(2,3), stride=(1,2)) -> None:
        super(DeconvGLU, self).__init__()
        
        self.main = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, output_padding=pad)
        self.gate = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, output_padding=pad)
    
    def forward(self, x):
        main = self.main(x)
        gate = self.gate(x)
        main = main[:,:,:-1,:] # pad on ConvTranspose2d
        gate = gate[:,:,:-1,:] # pad on ConvTranspose2d

        return main * torch.sigmoid(gate)
    

class Encoder_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(2,3)) -> None:
        super(Encoder_block, self).__init__()
        
        self.GLU_BN_PReLU_RSU = nn.Sequential(
            ConvGLU(in_ch, out_ch, kernel_size=kernel_size),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x):
        x = self.GLU_BN_PReLU_RSU(x)
        return x


class Decoder_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(2,3), pad=0, output_pad=(0,1)) -> None:
        super(Decoder_block, self).__init__()

        self.GLU_BN_PReLU_RSU = nn.Sequential(
            DeconvGLU(in_ch, out_ch, kernel_size=kernel_size, pad=pad),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(out_ch),
        )
    
    def forward(self, x):
        x = self.GLU_BN_PReLU_RSU(x)
        return x


class Res_block(nn.Module):
    def __init__(self, in_ch, kernel_size=(2,3), stride=(1,1)):
        super(Res_block, self).__init__()

        self.res_block = nn.Sequential(
            nn.ConstantPad2d((1,1,1,0), 0), # pad 0 on the start of Time axis
            nn.Conv2d(in_ch, in_ch, kernel_size, stride),
            nn.ReLU(inplace=True),
            nn.ConstantPad2d((1,1,1,0), 0), # pad 0 on the start of Time axis
            nn.Conv2d(in_ch, in_ch, kernel_size, stride),
        )
    def forward(self, x):
        x = x + self.res_block(x)
        return torch.relu(x)




class TriU_Net(nn.Module):
    def __init__(self, nfft=512, hid_ch=64) -> None:
        super(TriU_Net, self).__init__()
        self.nfft = nfft
        self.hid_ch = hid_ch

        # BF Stage, MB MVDR
        # Encoder in MB MVDR
        self.enc1 = nn.ModuleList()
        self.enc1.append(Encoder_block(18, hid_ch, kernel_size=(2,5)))
        self.enc1.append(Encoder_block(hid_ch, hid_ch))
        self.enc1.append(Encoder_block(hid_ch, hid_ch))
        self.enc1.append(Encoder_block(hid_ch, hid_ch))
        self.enc1.append(Encoder_block(hid_ch, hid_ch))
        self.enc1.append(Encoder_block(hid_ch, hid_ch))

        # Conformer in MB MVDR
        self.conformer1 = sConformer(input_dim=192,num_heads=4,ffn_dim=192*2,num_layers=2,depthwise_conv_kernel_size=31,dropout=0.1)

        # Decoder in MB MVDR
        self.dec1 = nn.ModuleList()
        self.dec1.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec1.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec1.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec1.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec1.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec1.append(Decoder_block(hid_ch*2, 1, kernel_size=(2,5), output_pad=0))
        
        # Mask Generator in MB MVDR
        self.lstm1 = nn.LSTM(input_size=nfft//2+1, hidden_size=hid_ch, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(hid_ch, nfft//2+1)

        self.mvdr = torchaudio.transforms.MVDR(ref_channel=0, solution='stv_power', multi_mask=False)
        
        # PF Stage
        # Encoder in PF
        self.enc2 = nn.ModuleList()
        self.enc2.append(Encoder_block(4, hid_ch, kernel_size=(2,5)))
        self.enc2.append(Encoder_block(hid_ch, hid_ch))
        self.enc2.append(Encoder_block(hid_ch, hid_ch))
        self.enc2.append(Encoder_block(hid_ch, hid_ch))
        self.enc2.append(Encoder_block(hid_ch, hid_ch))
        self.enc2.append(Encoder_block(hid_ch, hid_ch))

        # Conformer in PF
        self.conformer2 = sConformer(input_dim=192,num_heads=4,ffn_dim=192*2,num_layers=2,depthwise_conv_kernel_size=31,dropout=0.1)

        # Decoder in PF
        self.dec2 = nn.ModuleList()
        self.dec2.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec2.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec2.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec2.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec2.append(Decoder_block(hid_ch*2, hid_ch))
        self.dec2.append(Decoder_block(hid_ch*2, 2, kernel_size=(2,5), output_pad=0))
        
        # Mask Generator in PF
        self.lstm2 = nn.LSTM(input_size=nfft//2+1, hidden_size=hid_ch, num_layers=2, batch_first=True)
        self.linear2 = nn.Linear(hid_ch, nfft//2+1)

        # DC Stage
        # Decoder in DC
        self.dec3 = nn.ModuleList()
        self.dec3.append(Decoder_block(hid_ch*4, hid_ch))
        self.dec3.append(Decoder_block(hid_ch*3, hid_ch))
        self.dec3.append(Decoder_block(hid_ch*3, hid_ch))
        self.dec3.append(Decoder_block(hid_ch*3, hid_ch))
        self.dec3.append(Decoder_block(hid_ch*3, hid_ch))
        self.dec3.append(Decoder_block(hid_ch*3, 2, kernel_size=(2,5), output_pad=0))

        # ResBlock in DC
        res_list = []
        for i in range(3):
            res_list.append(Res_block(in_ch=4))
        self.res = nn.Sequential(*res_list)
        self.bottleneck = nn.Conv2d(4, 2, kernel_size=1)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
    
    def forward(self, x):
        '''
        x: [batch, ch, samples], Note that ch=9
        '''
        with torch.no_grad():
            b, ch, sample = x.shape

            # STFT and compress
            # x -> x: [b, ch, sample] -> [b, ch*2, f, t]
            x = torch.stft(
                x.view(b*ch, sample), n_fft=512, hop_length=128, window=torch.hann_window(512, device=x.device, dtype=torch.float), 
                return_complex=True)
            x = x.view(b, ch, 257, -1).transpose(-2,-1)
            x = self.compress_spec(x)

            x_residual = x.clone()
            b, ch, T, F = x.shape
            x_ref = torch.stack([x[:,0,:,:].real, x[:,0,:,:].imag], dim=1)
            x = torch.cat([x.real, x.imag], dim=1)

            # Enc1
            x_enc1 = []
            for i, enc in enumerate(self.enc1):
                x = enc(x)
                x_enc1.append(x)
            
            # Conformer1
            x = x.transpose(1,2).reshape([b, T, -1]) #[batch, ch, T, F] -> [batch, T, ch, F] -> [batch, ch*F, T]
            x, _ = self.conformer1(x, torch.ones(x.shape[0], device=x.device)*x.shape[1])
            x = x.reshape([b, T, self.hid_ch, -1]).transpose(1,2)
            x_conformer = x.clone()

            # Dec1
            for i, dec in enumerate(self.dec1):
                x = dec(torch.cat([x, x_enc1[-i-1]], dim=1))

            # Mask Generator
            x = x.squeeze(dim=1) # [B, 1, T, F] -> [B, T, F]
            x, (_,_) = self.lstm1(x)
            x = self.linear1(x)
            x = torch.sigmoid(x) 
            x = x.transpose(-2,-1) # [B, T, F] -> [B, F, T]
            # return x ## x denotes IRM
            
            # Do MVDR
            Y = self.mvdr(x_residual.transpose(-1,-2), x, 1-x).transpose(-1,-2)
            Y = torch.stack([Y.real, Y.imag], dim=1)  

        ## BFM
        # Enc2
        Y_residual = Y.clone()
        Y = torch.cat([Y, x_ref], dim=1)
        Y_enc2 = []
        for i, enc in enumerate(self.enc2):
            Y = enc(Y)
            Y_enc2.append(Y)
        
        # Conformer2
        Y = Y.transpose(1,2).reshape([b, T, -1]) #[batch, ch, T, F] -> [batch, T, ch, F] -> [batch, ch*F, T]
        Y, _ = self.conformer2(Y, torch.ones(Y.shape[0], device=Y.device)*Y.shape[1])
        Y = Y.reshape([b, T, self.hid_ch, -1]).transpose(1,2)
        Y_conformer = Y.clone()

        # Dec2
        for i, dec in enumerate(self.dec2):
            Y = dec(torch.cat([Y, Y_enc2[-i-1]], dim=1))
        
        # Mask Generator
        Y = Y.view([-1, T, self.nfft//2+1]) # [batch, ch, T, F] -> [batch*ch, T, F]
        Y, (_,_) = self.lstm2(Y)
        Y = Y.view([b, -1, T, self.hid_ch])
        Y = self.linear2(Y)

        Y_out = torch.empty_like(Y_residual)
        Y_out[:,0,:,:] = Y_residual[:,0,:,:] * Y[:,0,:,:] - Y_residual[:,1,:,:] * Y[:,1,:,:]
        Y_out[:,1,:,:] = Y_residual[:,0,:,:] * Y[:,1,:,:] + Y_residual[:,1,:,:] * Y[:,0,:,:]
        
        ## RRM
        # Dec3
        Y = torch.cat([Y_conformer, x_conformer], dim=1)
        for i, dec in enumerate(self.dec3):
            Y = dec(torch.cat([Y, Y_enc2[-i-1], x_enc1[-i-1]], dim=1))

        Y = torch.cat([Y, x_ref], dim=1)
        Y = self.res(Y)
        Y = self.bottleneck(Y)

        Y_out[:,0,:,:] = Y_out[:,0,:,:] + Y[:,0,:,:]
        Y_out[:,1,:,:] = Y_out[:,1,:,:] + Y[:,1,:,:]

        return Y_out

    
    def decompress_spec(self, x):
        '''
        x: [batch, 2, time, freq], float
        '''
        mag, phase = torch.norm(x, dim=1) ** 2, torch.atan2(x[:, 1,:,:], x[:, 0,:,:])
        x = torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=1)
        return x
    def compress_spec(self, x):
        '''
        x: [batch, ch, time, freq], complex
        '''
        mag, phase = x.abs(), x.angle()
        x = (mag**0.5) * torch.exp(1j*phase)
        return x


def get_Num_parameter(model):
    total = sum([param.nelement() for param in model.parameters()])
    trainable_num = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Total parameters: {total/1e6:.4f}M, Trainable: {trainable_num/1e6:.4f}M" )


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## 10s, 9-ch 16 kHz wav
    Batch = 4
    data = torch.rand([Batch, 9, 64000]).to(DEVICE)

    net = TriU_Net().to(DEVICE)
    get_Num_parameter(net)

    # infering
    net.eval()
    with torch.no_grad():
        out = net(data)
        out = net.uncompress_spec(out)
        print(f"input shape: {data.shape} \noutput shapt: {out.shape}")
        
        out = (out[:,0,...] + 1j*out[:,1,...]).transpose(-1,-2)
        enh_wav = torch.istft(out, n_fft=512, hop_length=128, window=torch.hann_window(512, device=out.device))
        print(f"enhanced wav shape: {enh_wav.shape}")

        ## write enhanced wav
        # import soundfile as sf
        # for i in range(B):
        # sf.write('path/to/your/enhanced_Wav.wav', enh_wav[i,:], 16000)













