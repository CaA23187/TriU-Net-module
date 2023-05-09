from statistics import mean
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import soundfile as sf
import os


from TriU_Net import TriU_Net 


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    noisy, fs = sf.read('wavs/9ULA_synth_kkl_babble_snr-5.wav')
    noisy = torch.FloatTensor(noisy).T.unsqueeze(0).to(DEVICE) ## [1, ch, samples]
    print("noisy shape: ", noisy.shape)

    net = TriU_Net().to(DEVICE)
    net.load_state_dict(torch.load('TriU_Net.pt', map_location=DEVICE),strict=True)
    net.eval()

    print('Enhancing...')
    with torch.inference_mode():
        out_spec = net(noisy)
        out_spec = net.decompress_spec(out_spec)
        
        out_spec = (out_spec[:,0,...] + 1j*out_spec[:,1,...]).transpose(-1,-2)
        enh_wav = torch.istft(out_spec, n_fft=512, hop_length=128, window=torch.hann_window(512, device=out_spec.device))
        print(f"enhanced wav shape: {enh_wav.shape}")

        enh_wav = enh_wav.squeeze(0).cpu().numpy() 
        sf.write('wavs/enhanced.wav', enh_wav, 16000)










    # json_path = "./json_file/"
    # test_dataset = MyDataset_HybridNBF_2(json_dir=json_path, flag='test_new_whole', info=True)
    # test_loader = DataLoader(dataset=test_dataset,
    #                           batch_size=BATCH_SIZE, 
    #                           shuffle=False,
    #                           pin_memory=True,
    #                           num_workers=16)

    # out_path = './out_wav/' + 'new_' + model_name
    # # out_path = './out_wav/' + 'online_MBMVDR_souden'
    # # out_path = './out_wav/' + 'HybridNBF_R'
    # print('out_path: ', out_path)
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    #     print('mkdir out_path')

    # net = TriU_Net().to(DEVICE)
    # net.load_state_dict(torch.load('TriU_Net', map_location=DEVICE),strict=True)
    #     # param = torch.load('weights/HybridNBF_online_mvdrnc_stagej_DS.pt', map_location=DEVICE)
    #     # # for key, _ in param.items():  # 遍历预训练权重的有序字典
    #     #     # del_key.append(key)
    #     #     # print(key)
    #     # del_key = ['mvdr.psd_s','mvdr.psd_n','mvdr.mask_sum_s','mvdr.mask_sum_n']
    #     # for key in del_key:
    #     #     del param[key]
    #     # net.load_state_dict(param, strict=False)
    # loss_func = SI_SNR()

    # ######################
    # # test the model #
    # ######################
    # net.eval() # prep model for evaluation
    # test_loader = tqdm(test_loader)

    # test_losses = []
    # mean_loss = 0
    # mean_sisnr_origin = 0
    # mean_sisnr_out = 0
    # idx = 0
    # pcnt = 0
    # p_out_avg = 0
    # p_ori_avg = 0
    # with torch.inference_mode():
    #     for step, (feats, labels, config) in enumerate(test_loader):
    #         feats = feats.to(DEVICE, non_blocking=True)
    #         labels = labels.to(DEVICE, non_blocking=True)

    #         out = net(feats)
    #         out = uncompress_spec(out)
    #         labels = uncompress_spec(labels)
    #         out = out.transpose(-1,-2)
    #         out = out[:,0,:,:] + 1j*out[:,1,:,:]
    #         out = torch.istft(out, n_fft=512, hop_length=128, window=torch.hann_window(512, device=labels.device))

    #         labels = labels.transpose(-1,-2)
    #         labels = labels[:,0,:,:] + 1j*labels[:,1,:,:]
    #         labels = torch.istft(labels, n_fft=512, hop_length=128, window=torch.hann_window(512, device=labels.device))

    #         loss = loss_func(out, labels)
    #         mean_loss = (mean_loss*step + loss.detach())/(step + 1)
    #         test_loader.desc = "mean loss {}".format(round(mean_loss.item(),5))
            
    #         feats = feats.cpu().numpy()
    #         out = out.cpu().numpy()
    #         labels = labels.cpu().numpy()
            
    #         for _ in range(feats.shape[0]):
    #             sir = config['SIR'][_]
    #             noisy_ratio = config['noisy_ratio'][_]
    #             sf.write('{}/{}_sir{:.2f}dB_noisyR{:.2f}_ch1.wav'.format(out_path, idx, sir, noisy_ratio), feats[_,0,:], samplerate=sr)
    #             sf.write('{}/{}_sir{:.2f}dB_noisyR{:.2f}_out.wav'.format(out_path, idx, sir, noisy_ratio), out[_,:], samplerate=sr)
    #             sf.write('{}/{}_sir{:.2f}dB_noisyR{:.2f}_tgt.wav'.format(out_path, idx, sir, noisy_ratio), labels[_,:], samplerate=sr)
    #             idx = idx+1
                

