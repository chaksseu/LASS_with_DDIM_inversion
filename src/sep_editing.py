import os
import sys

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(proj_dir, 'src_audioldm')
sys.path.extend([proj_dir, src_dir])

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt
import soundfile as sf
from src.audioldm import AudioLDM
from dataprocessor import AudioDataProcessor

def calculate_sdr(ref: np.ndarray, est: np.ndarray, eps=1e-10) -> float:
    r"""Calculate SDR between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    reference = ref
    noise = est - reference
    numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr

def calculate_sisdr(ref, est):
    r"""Calculate SDR between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    eps = np.finfo(ref.dtype).eps
    reference = ref.copy()
    estimate = est.copy()
    reference = reference.reshape(reference.size, 1)
    estimate = estimate.reshape(estimate.size, 1)
    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)
    e_true = a * reference
    e_res = estimate - e_true
    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()
    sisdr = 10 * np.log10((eps+ Sss)/(eps + Snn))
    return sisdr 

class Mask(nn.Module):
    def __init__(self, device, channel=1, height=1024, width=513):
        super().__init__()
        # self.weight = nn.Parameter(torch.randn((channel, height, width)))
        self.weight = nn.Parameter(torch.full((channel, height, width), 3.0))
        self.to(device)
        self.device = device
    def forward(self):
        return torch.sigmoid(self.weight)


def test_wav2mel2wav_reconstruction(wavpath, fname='000000.wav'):
    audioldm = AudioLDM('cuda:0')
    device = audioldm.device
    processor = AudioDataProcessor(device=device)

    wav = processor.read_wav_file(wavpath)
    
    wav_ = processor.prepare_wav(wav)
    mel = processor.wav_to_mel(wav_)  # [1,1,1024,512]
    recon_wav = processor.inverse_mel_with_phase(mel)

    sisdr = calculate_sisdr(wav, recon_wav)
    print(sisdr)
    recon_wav = recon_wav.squeeze()
    sf.write('./test/wav2mel2wav/000000.wav', recon_wav, 16000)

def test_wav2stft2wav_reconstruction(wavpath, fname='000000.wav'):
    audioldm = AudioLDM('cuda:0')
    device = audioldm.device
    processor = AudioDataProcessor(device=device)

    wav = processor.read_wav_file(wavpath)
    
    wav_ = processor.prepare_wav(wav)
    stft, stft_c = processor.wav_to_stft(wav_)  # [1,513,1024] [1,513,1024]
    recon_wav = processor.inverse_stft(stft, stft_c)

    sisdr = calculate_sisdr(wav, recon_wav)
    print(sisdr)
    recon_wav = recon_wav.squeeze()
    sf.write('./test/wav2stft2wav/000000.wav', recon_wav, 16000)

def inference(audioldm, processor, target_path, mixed_path, config):
    device = audioldm.device
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    batchsize = config['batchsize']
    strength = config['strength']
    iteration = config['iteration']
    text = config['text']

    for iter in range(iteration):
        target_wav = processor.read_wav_file(target_path)
        mixed_wav = processor.read_wav_file(mixed_path)

        sisdr = calculate_sisdr(ref=target_wav, est=mixed_wav)
        print(sisdr)

        # ------------------------------------------------------------------ #

        mixed_wav_ = processor.prepare_wav(mixed_wav)
        mixed_stft, mixed_stft_c = processor.wav_to_stft(mixed_wav_)
        mixed_mel = processor.wav_to_mel(mixed_wav_)

        mel_tar_samples = []
        for batch in tqdm(range(batchsize)):
            mel_sample = audioldm.edit_audio_with_ddim(
                                mel=mixed_mel,
                                text=text,
                                duration=10.24,
                                batch_size=1,
                                transfer_strength=strength,
                                guidance_scale=2.5,
                                ddim_steps=50,
                                clipping = False,
                                return_type="mel",
                            )
            
            mel_tar_samples.append(mel_sample) # [30, 1, 1024, 512]

            wav_sample = processor.inverse_mel_with_phase(mel_sample, mixed_stft_c)
            wav_sample = wav_sample.squeeze()
            sf.write(f'./test/sampling/edited_{text}_{strength}_{iter}_{batch}.wav', wav_sample, 16000)

        # mel_tar = torch.stack(mel_tar_samples).mean(dim=0)  # 평균 계산
        mel_tar = torch.cat(mel_tar_samples, dim=0)

        # ------------------------------------------------------------------ #

        sisdrs_list = []
        sdris_list = []
        loss_values = []

        mask = Mask(device, 1, 513, 1024)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(mask.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):

            optimizer.zero_grad()  # 그래디언트 초기화
            masked_stft = (mixed_stft - mixed_stft.min()) * mask() + mixed_stft.min()  #ts[1,513,1024]
            
            masked_mel = processor.masked_stft_to_masked_mel(masked_stft)  # [1,1,1024,512]
            masked_mel_expended = masked_mel.repeat(batchsize, 1, 1, 1)

            loss = criterion(mel_tar, masked_mel_expended)  # 손실 계산

            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

            loss_values.append(loss.item())  # 손실값 저장

            ##
            wav_sep = processor.inverse_stft(masked_stft, mixed_stft_c)
            target_wav = target_wav.squeeze(0)
            mixed_wav = mixed_wav.squeeze(0)
            assert len(wav_sep) <= len(target_wav), (len(wav_sep), len(target_wav))
            wav_src = target_wav[:len(wav_sep)]
            wav_mix = mixed_wav[:len(wav_sep)]

            sdr_no_sep = calculate_sdr(ref=wav_src, est=wav_mix)
            sdr = calculate_sdr(ref=wav_src, est=wav_sep)
            sdri = sdr - sdr_no_sep
            sisdr = calculate_sisdr(ref=wav_src, est=wav_sep)

            sisdrs_list.append(sisdr)
            sdris_list.append(sdri)
            target_wav = np.expand_dims(target_wav, axis=0)
            mixed_wav = np.expand_dims(mixed_wav, axis=0)
            ##

            # if epoch % 1000 == 0:
            #     print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
            #     print(f"sisdr: {sisdr:.4f}, sdri: {sdri:.4f}")
                
        # ------------------------------------------------------------------ #

        # plt.plot(loss_values)
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Loss Trend')
        # plt.savefig(f'./test/plot/loss_{text}_{iter}.png')
        # plt.close()

        plt.plot(sisdrs_list)
        plt.xlabel('Epoch')
        plt.ylabel('sisdr')
        plt.title('sisdr Trend')
        plt.savefig(f'./test/plot/sisdr_{text}_{iter}.png')
        plt.close()

        plt.plot(sdris_list)
        plt.xlabel('Epoch')
        plt.ylabel('sdri')
        plt.title('sdri Trend')
        plt.savefig(f'./test/plot/sdri_{text}_{iter}.png')
        plt.close()

        wav_sep = processor.inverse_stft(masked_stft, mixed_stft_c)
        wav_sep = wav_sep.squeeze()
        sf.write(f'./test/sep_{text}_{iter}.wav', wav_sep, 16000)

        mixed_path = f'./test/sep_{text}_{iter}.wav'
        print(f"iteration: {iter} // sisdr: {sisdrs_list[-1]:.4f}, sdri: {sdris_list[-1]:.4f}")

    print(f"Final: sample: {text}\nFFF sisdr: {sisdrs_list[-1]:.4f}, sdri: {sdris_list[-1]:.4f}")
    return sisdrs_list[-1], sdris_list[-1]


if __name__ == "__main__":

    audioldm = AudioLDM('cuda:0')
    device = audioldm.device
    processor = AudioDataProcessor(device=device)
    target_path = './samples/A_cat_meowing.wav'
    mixed_path = './samples/a_cat_n_stepping_wood.wav'

    config = {
        'text': 'A cat meowing',
        'num_epochs': 1000,
        'batchsize': 3,
        'strength': 0.5,
        'learning_rate': 0.01,
        'iteration': 1
    }

    inference(audioldm, processor, target_path, mixed_path, config)
    
