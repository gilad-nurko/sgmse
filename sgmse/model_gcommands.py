import time
from math import ceil
import warnings

import torch
import pytorch_lightning as pl
import torch.distributed as dist
import torchaudio
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from librosa import resample
import torchmetrics
import whisper

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss


class WhisperGuidedScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="score_matching", help="The type of loss function to use.")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2", help="The weighting of the loss function.")
        parser.add_argument("--network_scaling", type=str, default=None, help="The type of loss scaling to use.")
        parser.add_argument("--c_in", type=str, default="1", help="The input scaling for x.")
        parser.add_argument("--c_out", type=str, default="1", help="The output scaling.")
        parser.add_argument("--c_skip", type=str, default="0", help="The skip connection scaling.")
        parser.add_argument("--sigma_data", type=float, default=0.1, help="The data standard deviation.")
        parser.add_argument("--l1_weight", type=float, default=0.001, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--pesq_weight", type=float, default=0.0, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--sr", type=int, default=16000, help="The sample rate of the audio files.")
        parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model variant to use (tiny, base, small, medium, large)")
        parser.add_argument("--lang", type=str, default="en", help="Language for Whisper decoding")
        parser.add_argument("--debug", type=bool, default=False, help="Whether to enable debug visualization during validation")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='score_matching', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, 
        l1_weight=0.001, pesq_weight=0.0, sr=16000, whisper_lang='en', data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize WER metric
        self.wer_metric = torchmetrics.WordErrorRate()
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.l1_weight = l1_weight
        self.pesq_weight = pesq_weight
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.sr = sr
        # Initialize PESQ loss if pesq_weight > 0.0
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=['no_wandb'])

        self.allowed_words = ["down","go","left","no","off","on","right","stop","up","yes"]
        self.options = whisper.DecodingOptions(language=whisper_lang, without_timestamps=True)
        self.whisper = whisper.load_model(f'base')
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=whisper_lang, task=self.options.task)
        if hasattr(self.whisper, 'alignment_heads') and self.whisper.alignment_heads.is_sparse:
        # Convert the sparse buffer to a dense buffer
            self.whisper.alignment_heads = self.whisper.alignment_heads.to_dense()    
        # Freeze Whisper parameters but keep gradient computation enabled
        for param in self.whisper.parameters():
            param.requires_grad = False
            
            # Cross entropy loss for ASR
            self.whisper_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        ids = [self.tokenizer.encode(" " + w) for w in self.allowed_words]
        for w, tok in zip(self.allowed_words, ids):
            if len(tok) != 1:
                print(f"⚠️  '{w}' tokenises to {tok} – handle multi-token keywords!")
        self.allowed_toks = torch.tensor([tok[0] for tok in ids])
        self.debug = False
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)

    # def _loss(self, forward_out, x_t, z, t, mean, x, masks):
    #     """
    #     Different loss functions can be used to train the score model, see the paper: 
        
    #     Julius Richter, Danilo de Oliveira, and Timo Gerkmann
    #     "Investigating Training Objectives for Generative Speech Enhancement"
    #     https://arxiv.org/abs/2409.10753

    #     """

    #     sigma = self.sde._std(t)[:, None, None, None]

    #     if self.loss_type == "score_matching":
    #         score = forward_out
    #         if self.loss_weighting == "sigma^2":
    #             losses = torch.square(torch.abs(score * sigma + z)) # Eq. (7)
    #         else:
    #             raise ValueError("Invalid loss weighting for loss_type=score_matching: {}".format(self.loss_weighting))
    #         # Sum over spatial dimensions and channels and mean over batch
    #         loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
    #     elif self.loss_type == "denoiser":
    #         score = forward_out
    #         D = score * sigma.pow(2) + x_t # equivalent to Eq. (10)
    #         losses = torch.square(torch.abs(D - mean)) # Eq. (8)
    #         if self.loss_weighting == "1":
    #             losses = losses
    #         elif self.loss_weighting == "sigma^2":
    #             losses = losses * sigma**2
    #         elif self.loss_weighting == "edm":
    #             losses = ((sigma**2 + self.sigma_data**2)/((sigma*self.sigma_data)**2))[:, None, None, None] * losses
    #         else:
    #             raise ValueError("Invalid loss weighting for loss_type=denoiser: {}".format(self.loss_weighting))
    #         # Sum over spatial dimensions and channels and mean over batch
    #         loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))     
    #     elif self.loss_type == "data_prediction":
    #         x_hat = forward_out
    #         B, C, F, T = x.shape

    #         # losses in the time-frequency domain (tf)
    #         losses_tf = (1/(F*T))*torch.square(torch.abs(x_hat - x))
    #         losses_tf = torch.mean(0.5*torch.sum(losses_tf.reshape(losses_tf.shape[0], -1), dim=-1))

    #         # losses in the time domain (td)
    #         target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
    #         x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
    #         x_td = self.to_audio(x.squeeze(), target_len)
    #         losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
    #         losses_l1 = torch.mean(0.5*torch.sum(losses_l1.reshape(losses_l1.shape[0], -1), dim=-1))

    #         # losses using PESQ
    #         if self.pesq_weight > 0.0:
    #             losses_pesq = self.pesq_loss(x_td, x_hat_td)
    #             losses_pesq = torch.mean(losses_pesq)
    #             # combine the losses
    #             loss = losses_tf + self.l1_weight * losses_l1 + self.pesq_weight * losses_pesq 
    #         else:
    #             loss = losses_tf + self.l1_weight * losses_l1
    #     else:
    #         raise ValueError("Invalid loss type: {}".format(self.loss_type))

    #     return loss

    def _loss(self, forward_out, x_t, z, t, mean, x, masks):
        """
        Masks: (B, 2, F, T) with 1 for valid frames, 0 for padding.
            masks[:, 1] corresponds to the clean target 'x'.
        """

        # ----- helpers -----
        def masked_mean(tensor, mask, sum_dims):
            # tensor, mask broadcastable; mask is 0/1
            num = (tensor * mask).sum(dim=sum_dims)
            den = mask.sum(dim=sum_dims).clamp_min(1e-8)
            return (num / den).mean()  # mean over batch

        B, C, F, T = x.shape
        device = x.device

        # clean TF mask (broadcast to channel dim)
        # masks_x: (B, 1, F, T) with 0/1 floats
        masks_x = masks[:, 1:2, ...].to(x.dtype)

        sigma = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            score = forward_out
            if self.loss_weighting == "sigma^2":
                # Eq. (7)
                losses = torch.square(torch.abs(score * sigma + z))  # (B, C, F, T)
            else:
                raise ValueError(f"Invalid loss weighting for loss_type=score_matching: {self.loss_weighting}")

            # masked mean over C,F,T then mean over batch; keep your 0.5 factor
            loss = 0.5 * masked_mean(losses, masks_x, sum_dims=(1, 2, 3))

        elif self.loss_type == "denoiser":
            score = forward_out
            # Eq. (10)
            D = score * sigma.pow(2) + x_t
            # Eq. (8)
            losses = torch.square(torch.abs(D - mean))  # (B, C, F, T)

            if self.loss_weighting == "1":
                pass
            elif self.loss_weighting == "sigma^2":
                losses = losses * sigma**2
            elif self.loss_weighting == "edm":
                w = ((sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data)**2))[:, None, None, None]
                losses = w * losses
            else:
                raise ValueError(f"Invalid loss weighting for loss_type=denoiser: {self.loss_weighting}")

            loss = 0.5 * masked_mean(losses, masks_x, sum_dims=(1, 2, 3))

        elif self.loss_type == "data_prediction":
            x_hat = forward_out

            # ----- TF loss (masked) -----
            losses_tf = torch.square(torch.abs(x_hat - x))  # (B, C, F, T)
            # original code averaged by (F*T); here we do a true masked mean
            loss_tf = 0.5 * masked_mean(losses_tf, masks_x, sum_dims=(1, 2, 3))

            # ----- TD loss (masked on valid samples only) -----
            # Build a time mask in samples from the TF mask (1 if any freq is valid)
            # time_mask: (B, T_frames)
            time_mask_frames = masks[:, 1, ...].any(dim=1).to(x.dtype)  # (B, T)
            # valid samples per item (approx): frames * hop_length
            hop = self.data_module.hop_length
            valid_len = (time_mask_frames.sum(dim=1) * hop).long()  # (B,)

            target_len = (self.data_module.num_frames - 1) * hop
            x_hat_td = self.to_audio(x_hat.squeeze(1), target_len)  # (B, target_len)
            x_td     = self.to_audio(x.squeeze(1),    target_len)   # (B, target_len)

            # Build per-sample 1D masks in samples: (B, target_len)
            time_mask_samples = torch.zeros((B, target_len), dtype=x.dtype, device=device)
            for b in range(B):
                L = int(valid_len[b].item())
                L = min(L, target_len)
                if L > 0:
                    time_mask_samples[b, :L] = 1.0

            # L1 masked mean per sample, then mean over batch (keep your 0.5 factor)
            l1 = torch.abs(x_hat_td - x_td)  # (B, target_len)
            num = (l1 * time_mask_samples).sum(dim=1)
            den = time_mask_samples.sum(dim=1).clamp_min(1e-8)
            loss_l1 = 0.5 * (num / den).mean()

            # ----- optional PESQ on the valid (unpadded) part only -----
            if self.pesq_weight > 0.0:
                pesq_list = []
                for b in range(B):
                    L = int(valid_len[b].item())
                    L = max(L, 1)  # avoid zero length
                    ref = x_td[b, :L].unsqueeze(0)
                    est = x_hat_td[b, :L].unsqueeze(0)
                    pesq_list.append(self.pesq_loss(ref, est))  # scalar tensor
                losses_pesq = torch.stack(pesq_list).mean()
                loss = loss_tf + self.l1_weight * loss_l1 + self.pesq_weight * losses_pesq
            else:
                loss = loss_tf + self.l1_weight * loss_l1

        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")

        return loss


    def _step(self, batch, batch_idx):
        specs, masks, multilingual_tokens, labels = batch["specs"], batch["masks"], batch["multilingual_tokens"], batch["labels"]
        y = specs[:, 0:1, ...]   # (B,1,F,T)  noisy
        x = specs[:, 1:2, ...]   # (B,1,F,T)  clean
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, y, t)
        z = torch.randn_like(x)  # i.i.d. normal distributed with var=0.5
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z
        forward_out = self(x_t, y, t)
        loss = self._loss(forward_out, x_t, z, t, mean, x, masks)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # ---- once-per-epoch gate ----
        if batch_idx == 0 and self.num_eval_files != 0:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            eval_files_per_gpu = self.num_eval_files // world_size
            indices_all = list(range(min(self.num_eval_files, len(self.data_module.valid_set))))
            if rank == world_size - 1:
                indices = indices_all[rank * eval_files_per_gpu :]
            else:
                indices = indices_all[rank * eval_files_per_gpu : (rank + 1) * eval_files_per_gpu]

            # accumulators
            pesq_sum = 0.0; si_sdr_sum = 0.0; estoi_sum = 0.0
            preds_clean, preds_noisy, preds_enh, refs = [], [], [], []

            vp = self.data_module.valid_set
            hop = self.data_module.hop_length
            n_fft = self.data_module.n_fft
            target_len = (self.data_module.num_frames - 1) * hop
            sr_target = getattr(self.data_module, "sample_rate", 16000)
            norm_mode = getattr(self.data_module, "normalize", "noisy")

            # ---------- closed-set Whisper decode helpers ----------
            special_ids = {
                self.tokenizer.eot,
                self.tokenizer.encode("<|en|>",           allowed_special={"<|en|>"})[0],
                self.tokenizer.encode("<|transcribe|>",   allowed_special={"<|transcribe|>"})[0],
                self.tokenizer.encode("<|notimestamps|>", allowed_special={"<|notimestamps|>"})[0],
                self.tokenizer.encode("<|nospeech|>",     allowed_special={"<|nospeech|>"})[0],
            }
            whisper_device = next(self.whisper.parameters()).device

            def keyword_index(labels_tensor: torch.Tensor) -> int:
                for j, t in enumerate(labels_tensor.tolist()):
                    if t not in special_ids:
                        return j
                # fallback: last token before EOT
                ids = labels_tensor.tolist()
                eot = self.tokenizer.eot
                return max(0, (ids.index(eot) if eot in ids else len(ids)) - 1)

            @torch.no_grad()  # eval‑only; remove if you later want gradients through Whisper
            def run_whisper_closedset(mel_80x3000, multilingual_tokens, labels):
                mel_80x3000 = mel_80x3000.to(whisper_device)
                multilingual_tokens = multilingual_tokens.to(whisper_device)
                labels = labels.to(whisper_device)
                feats  = self.whisper.encoder(mel_80x3000.unsqueeze(0))               # (1, D, T')
                logits = self.whisper.decoder(multilingual_tokens.unsqueeze(0), feats).squeeze(0)  # (L, V)
                kidx   = keyword_index(labels)
                kw_logits = logits[kidx, self.allowed_toks.to(logits.device)]         # (10,)
                pred_id   = int(torch.argmax(kw_logits).item())
                return self.allowed_words[pred_id]

            # ---------- STFT -> log‑mel helper (inline, differentiable) ----------
            # Build a mel filterbank once for this block.
            n_freqs = n_fft // 2 + 1
            n_mels = 80
            try:
                # Use Whisper's exact filterbank if available (constant matrix, OK for autograd)
                from whisper.audio import mel_filters as whisper_mel_filters
                mel_fb = whisper_mel_filters(sr_target)[:n_mels, :]  # (80, n_freqs expected 201)
                if mel_fb.shape[1] != n_freqs:
                    # fallback if shapes don't match current n_fft
                    mel_fb = torchaudio.functional.melscale_fbanks(
                        n_freqs=n_freqs, n_mels=n_mels, sample_rate=sr_target,
                        f_min=0.0, f_max=sr_target/2, norm="slaney", mel_scale="slaney"
                    )
            except Exception:
                mel_fb = torchaudio.functional.melscale_fbanks(
                    n_freqs=n_freqs, n_mels=n_mels, sample_rate=sr_target,
                    f_min=0.0, f_max=sr_target/2, norm="slaney", mel_scale="slaney"
                )
            mel_fb = mel_fb.to(self.device)  # (80, n_freqs)

            def stft_to_logmel(spec_cplx_FT: torch.Tensor) -> torch.Tensor:
                """
                spec_cplx_FT: (F, T) complex from torch.stft
                returns: (80, 3000) log10-mel (pad/trim on time to 3000)
                """
                power = spec_cplx_FT.abs().pow(2.0)                       # (F, T)
                mel = mel_fb @ power                                      # (80, T)
                logmel = torch.log10(mel.clamp_min(1e-10))                # (80, T)
                T = logmel.size(-1)
                if T > 3000:
                    logmel = logmel[..., :3000]
                elif T < 3000:
                    logmel = F.pad(logmel, (0, 3000 - T))
                return logmel

            # ======================= MAIN LOOP =======================
            for i in indices:
                # dataset sample (for mask/tokens/labels)
                specs, masks, multilingual_tokens, labels = vp[i]  # specs: (2, F, T) complex
                # order: [noisy, clean]
                noisy_spec_fwd, clean_spec_fwd = specs[0].to(self.device), specs[1].to(self.device)

                # load waveforms by index (for audio-domain metrics)
                clean_path = vp.clean_files[i]
                noisy_path = vp.noisy_files[i]
                x, sr_x = torchaudio.load(clean_path)
                y, sr_y = torchaudio.load(noisy_path)
                if sr_x != sr_target:
                    x = torchaudio.functional.resample(x, sr_x, sr_target)
                if sr_y != sr_target:
                    y = torchaudio.functional.resample(y, sr_y, sr_target)

                def fit_to_target(sig):
                    cur = sig.size(-1)
                    pad = target_len - cur
                    if pad > 0:
                        sig = F.pad(sig, (0, pad))
                    elif pad < 0:
                        sig = sig[..., :target_len]
                    return sig
                x, y = fit_to_target(x), fit_to_target(y)

                # dataset-style normalization (single place)
                if norm_mode == "noisy":
                    norm = y.abs().max()
                elif norm_mode == "clean":
                    norm = x.abs().max()
                else:
                    norm = x.new_tensor(1.0)
                x, y = x / norm, y / norm

                # valid (unpadded) length for audio metrics
                n_valid_frames = int(masks[1].any(dim=0).sum().item())  # clean mask
                valid_len = min(n_valid_frames * hop, target_len)

                # -------- enhance -> spectrogram (no (de)norm inside) --------
                x_hat_spec_fwd, T_orig = self.enhance(y, N=self.sde.N, snr=0.33)    # (C,F,T) in forward/feature space
                x_hat_spec_fwd = x_hat_spec_fwd.squeeze(0).to(self.device)  # (F,T) if your C==1

                # -------- AUDIO METRICS (unchanged) --------
                x_hat_td = self.to_audio(x_hat_spec_fwd, T_orig).unsqueeze(0)  # (1, T_orig)
                x_valid     = x[..., :valid_len].squeeze(0)
                x_hat_valid = x_hat_td[..., :valid_len].squeeze(0)

                if sr_target != 16000:
                    x16    = torchaudio.functional.resample(x_valid.unsqueeze(0),    sr_target, 16000).squeeze(0)
                    xhat16 = torchaudio.functional.resample(x_hat_valid.unsqueeze(0), sr_target, 16000).squeeze(0)
                else:
                    x16, xhat16 = x_valid, x_hat_valid

                pesq_sum  += pesq(16000, x16.cpu().numpy(), xhat16.cpu().numpy(), 'wb')
                si_sdr_sum += si_sdr(x_valid.cpu().numpy(), x_hat_valid.cpu().numpy())
                estoi_sum += stoi(x_valid.cpu().numpy(), x_hat_valid.cpu().numpy(), sr_target, extended=True)

                # -------- WER via Whisper, using STFT -> log-mel (no whisper.log_mel_spectrogram) --------
                # Convert the *forward-transformed* specs back to linear STFT before mel
                noisy_lin  = self.data_module.spec_back(noisy_spec_fwd)        # (F,T) complex
                clean_lin  = self.data_module.spec_back(clean_spec_fwd)        # (F,T) complex
                enh_lin    = self.data_module.spec_back(x_hat_spec_fwd)        # (F,T) complex

                y_mel    = stft_to_logmel(noisy_lin)   # (80,3000)
                x_mel    = stft_to_logmel(clean_lin)   # (80,3000)
                xhat_mel = stft_to_logmel(enh_lin)     # (80,3000)

                pred_clean    = run_whisper_closedset(x_mel,    multilingual_tokens, labels)
                pred_noisy    = run_whisper_closedset(y_mel,    multilingual_tokens, labels)
                pred_enhanced = run_whisper_closedset(xhat_mel, multilingual_tokens, labels)

                # reference word (first non‑special token)
                kidx = keyword_index(labels)
                ref_tok = labels[kidx].item()
                pos = (self.allowed_toks.cpu() == ref_tok).nonzero(as_tuple=True)[0]
                ref_word = self.allowed_words[int(pos[0].item())] if len(pos) else self.tokenizer.decode([ref_tok]).strip()

                preds_clean.append(pred_clean)
                preds_noisy.append(pred_noisy)
                preds_enh.append(pred_enhanced)
                refs.append(ref_word)

            # ---- aggregate & log ----
            denom = len(indices)
            pesq_avg   = pesq_sum  / denom
            si_sdr_avg = si_sdr_sum/ denom
            estoi_avg  = estoi_sum / denom

            wer_clean     = self.wer_metric(preds_clean, refs)
            wer_corrupted = self.wer_metric(preds_noisy, refs)
            wer_enhanced  = self.wer_metric(preds_enh,  refs)

            self.log('pesq',          pesq_avg,   on_step=False, on_epoch=True, sync_dist=True)
            self.log('si_sdr',        si_sdr_avg, on_step=False, on_epoch=True, sync_dist=True)
            self.log('estoi',         estoi_avg,  on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_clean',     wer_clean,     on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_corrupted', wer_corrupted, on_step=False, on_epoch=True, sync_dist=True)
            self.log('wer_enhanced',  wer_enhanced,  on_step=False, on_epoch=True, sync_dist=True)

        # regular per-batch loss
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss



    def forward(self, x_t, y, t):
        """
        The model forward pass. In [1] and [2], the model estimates the score function. In [3], the model estimates 
        either the score function or the target data for the Schrödinger bridge (loss_type='data_prediction').
        
        [1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, and  Timo Gerkmann 
            "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models"
            IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023. 

        [2] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann
            "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation"
            ISCA Interspecch, Kos, Greece, Sept. 2024. 

        [3] Julius Richter, Danilo de Oliveira, and Timo Gerkmann
            "Investigating Training Objectives for Generative Speech Enhancement"
            https://arxiv.org/abs/2409.10753

        """

        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            F = self.dnn(self._c_in(t) * x_t, self._c_in(t) * y, t)
            
            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self.sde._std(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]

            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * x_t + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self.sde._std(t)[:, None, None, None]
                score = (F - x_t) / sigmas.pow(2)
                return score
            elif self.loss_type == 'data_prediction':
                x_hat = self._c_skip(t) * x_t + self._c_out(t) * F
                return x_hat
                
        # In [1] and [2], we use the old code:
        else:
            dnn_input = torch.cat([x_t, y], dim=1)            
            score = -self.dnn(dnn_input, t)
            return score

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self.sde._std(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_in type: {}".format(self.c_in))
    
    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self.sde._std(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self.sde._std(t)[:, None, None, None] 
        elif self.c_out == "edm":
            sigma = self.sde._std(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_out type: {}".format(self.c_out))
    
    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self.sde._std(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_skip type: {}".format(self.c_skip))

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(
    self, y, sampler_type="pc", predictor="reverse_diffusion",
    corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False, **kwargs
    ):
        """
        Takes a normalized waveform y (1, T), returns ENHANCED SPECTROGRAM (no iSTFT, no (de)normalization).
        """
        import time
        start = time.time()

        device = next(self.parameters()).device
        y = y.to(device)                     # (1, T)
        T_orig = y.size(-1)

        # --- NO normalization here; y is assumed already normalized ---

        # STFT -> forward transform -> pad for sampler
        Y = self._stft(y)                    # complex STFT
        Y = self._forward_transform(Y)       # model's spec domain
        Y = Y.unsqueeze(0)                   # (B=1, C, F, T)
        Y = pad_spec(Y)

        # ----- sampling -----
        if self.sde.__class__.__name__ == 'OUVESDE':
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(
                    predictor, corrector, Y, N=N,
                    corrector_steps=corrector_steps, snr=snr,
                    intermediate=False, **kwargs
                )
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y, N=N, **kwargs)
            else:
                raise ValueError(f"Invalid sampler type: {self.sde.sampler_type}")
        elif self.sde.__class__.__name__ == 'SBVESDE':
            sampler = self.get_sb_sampler(sde=self.sde, y=Y, sampler_type=self.sde.sampler_type)
        else:
            raise ValueError(f"Invalid SDE type: {self.sde.__class__.__name__}")

        sample, nfe = sampler()              # (1, C, F, T)
        x_hat_spec = sample.squeeze(0)       # (C, F, T)

        if timeit:
            sr = getattr(self.data_module, "sample_rate", 16000)
            rtf = (time.time() - start) / (T_orig / sr)
            return x_hat_spec, T_orig, nfe, rtf
        else:
            return x_hat_spec, T_orig

