import math
from typing import Union, List, Tuple, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from diffusers_3d.models.pvcnn import pvcnn_base, pvcnn_debug
from diffusers_3d.structures.points import PointTensor
from diffusers_3d.schedulers.ddpm import DDPMScheduler


import numpy as np
import matplotlib.pyplot as plt

def visualize_pointcloud_batch(path, pointclouds, pred_labels, labels, categories, vis_label=False, target=None,  elev=30, azim=225):
    batch_size = len(pointclouds)
    fig = plt.figure(figsize=(20,20))

    ncols = int(np.sqrt(batch_size))
    nrows = max(1, (batch_size-1) // ncols+1)
    for idx, pc in enumerate(pointclouds):
        if vis_label:
            label = categories[labels[idx].item()]
            pred = categories[pred_labels[idx]]
            colour = 'g' if label == pred else 'r'
        elif target is None:

            colour = 'g'
        else:
            colour = target[idx]
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=5)
        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')
        if vis_label:
            ax.set_title('GT: {0}\nPred: {1}'.format(label, pred))

    plt.savefig(path)
    plt.close(fig)


class GaussianDiffusion:
    def __init__(self,betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):
        model_output = denoise_fn(data, t)

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    ''' samples '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """
        Sample from the model
        """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                                 return_pred_xstart=True)
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(self, denoise_fn, shape, device,
                      noise_fn=torch.randn, clip_denoised=True, keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps

        """

        assert isinstance(shape, (tuple, list))
        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t,t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t.shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq,
                                 noise_fn=torch.randn,clip_denoised=True, keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps =  self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0,total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised,
                                  return_pred_xstart=False)
            if t % freq == 0 or t == total_steps-1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """
        Training loss calculation
        """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start.shape, dtype=data_start.dtype, device=data_start.device)
        assert noise.shape == data_start.shape and noise.dtype == data_start.dtype

        data_t = self.q_sample(x_start=data_start, t=t, noise=noise)

        if self.loss_type == 'mse':
            # predict the noise instead of x_start. seems to be weighted naturally like SNR
            eps_recon = denoise_fn(data_t, t)
            assert data_t.shape == data_start.shape
            assert eps_recon.shape == torch.Size([B, D, N])
            assert eps_recon.shape == data_start.shape
            losses = ((noise - eps_recon)**2).mean(dim=list(range(1, len(data_start.shape))))
        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
               + (mean1 - mean2)**2 * torch.exp(-logvar2))


class PVCNN(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embed_dim: int = 64,
        learning_rate: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        norm_weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        max_epochs: int = 60,
        warmup_iters: int = 500,
        optimizer_type: str = "Adam",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.center_input_sample = center_input_sample

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.norm_weight_decay = norm_weight_decay
        self.label_smoothing = label_smoothing
        self.max_epochs = max_epochs
        self.warmup_iters = warmup_iters
        self.optimizer_type = optimizer_type

        self.model = pvcnn_debug(in_channels, time_embed_dim)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False,
        )

        import numpy as np
        betas = np.linspace(0.0001, 0.02, 1000)

        self.diffusion = GaussianDiffusion(betas, "mse", "eps", "fixedsmall")

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
    ):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
        """
        # center input if necessary
        if self.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        # if not torch.is_tensor(timesteps):
        #     timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        # elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        #     timesteps = timesteps[None].to(sample.device)

        # # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        # timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        points = PointTensor(
            coords=sample[:, :, :3],
            features=sample.permute(0, 2, 1),
            is_channel_last=False,
            timesteps=timesteps,
        )

        return self.model(points).permute(0, 2, 1)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        assert noise is not None
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def training_step(self, batch, batch_idx: int):
        pcs, labels = batch.pcs, batch.labels

        batch_size = pcs.shape[0]

        # Sample noise that we'll add to the point clouds
        noise = torch.randn_like(pcs)
        # Sample a random timestep for each point cloud
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (batch_size,),
            dtype=torch.long, device=pcs.device,
        )

        # Add noise to the clean pcs according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # noisy_pcs = self.noise_scheduler.add_noise(pcs, noise, timesteps)
        noisy_pcs = self.q_sample(x_start=pcs, t=timesteps, noise=noise)

        # Predict the noise residual
        model_output = self.forward(noisy_pcs, timesteps)

        prediction_type = self.noise_scheduler.prediction_type
        if prediction_type == "epsilon":
            loss = F.mse_loss(model_output, noise)
            # pcs = pcs.permute(0, 2, 1)
            # noise = noise.permute(0, 2, 1)
            # model_output = model_output.permute(0, 2, 1)
            # loss = ((noise - model_output)**2).mean(dim=list(range(1, len(pcs.shape)))).mean()
        elif prediction_type == "sample":
            raise NotImplementedError(prediction_type)
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.get_bpd(pcs)

        # self.log("train/bpd_b", total_bpd_b, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/vals_bt", vals_bt, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/prior_bpd_b", prior_bpd_b, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/mse_bt", mse_bt, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs):
        if (self.current_epoch + 1) % 10 != 0:
            return

        self.eval()

        print("vis")

        def _denoise(data, t):
            B, D,N= data.shape
            assert data.dtype == torch.float
            assert t.shape == torch.Size([B]) and t.dtype == torch.int64

            points = PointTensor(
                coords=data[:, :3, :].permute(0, 2, 1),
                features=data,
                is_channel_last=False,
                timesteps=t,
            )

            out = self.model(points)

            assert out.shape == torch.Size([B, D, N])
            return out

        with torch.no_grad():
            x_gen_eval = self.diffusion.p_sample_loop(
                _denoise,
                shape=(25, 3, 2048),
                device=self.device,
                noise_fn=torch.randn,
                clip_denoised=False,
            )

            visualize_pointcloud_batch(
                f"output/test2/epoch_{self.current_epoch:03d}.png",
                x_gen_eval.transpose(1, 2), None, None, None
            )

        self.train()

    def on_after_backward(self):
        for name, p in self.named_parameters():
            if p.grad is None:
                print(name)

        param_norm = torch.sqrt(sum(torch.sum(p ** 2) for p in self.parameters()))
        grad_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in self.parameters() if p.grad is not None))

        self.log("train/param_norm", param_norm, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=True, prog_bar=True)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, data, t, clip_denoised: bool, return_pred_xstart: bool):
        model_output = self.forward(data, t)

        self.model_var_type = "fixedsmall"
        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device), self.posterior_log_variance_clipped.to(data.device)),
            }[self.model_var_type]
            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(data)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        self.model_mean_type = "eps"
        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -.5, .5)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data, t=t)
        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    def get_vb_terms_bpd(
        self,
        samples: torch.Tensor,
        noisy_samples: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = False,
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=samples, x_t=noisy_samples, t=timesteps
        )
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            data=noisy_samples,
            t=timesteps,
            clip_denoised=clip_denoised,
            return_pred_xstart=True
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(samples.shape)))) / math.log(2.)

        return kl, pred_xstart

    @torch.no_grad()
    def _prior_bpd(self, x_start):
        B, T = x_start.shape[0], 1000
        t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T-1)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
        assert kl_prior.shape == x_start.shape
        return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / math.log(2.)

    @torch.no_grad()
    def get_bpd(self, samples: torch.Tensor):
        batch_size = samples.shape[0]

        vals_bt_ = torch.zeros([batch_size, 1000], device=samples.device)
        mse_bt_ = torch.zeros([batch_size, 1000], device=samples.device)
        for t in reversed(range(1000)):
            timesteps = torch.full(
                (batch_size,), fill_value=t, dtype=torch.long, device=samples.device
            )

            noise = torch.randn_like(samples)
            noisy_pcs = self.noise_scheduler.add_noise(samples, noise, timesteps)

            # Calculate VLB term at the current timestep
            new_vals_b, pred_xstart = self.get_vb_terms_bpd(
                samples, noisy_pcs, timesteps, clip_denoised=True
            )
            # MSE for progressive prediction loss
            assert pred_xstart.shape == samples.shape
            new_mse_b = ((pred_xstart-samples)**2).mean(dim=list(range(1, len(samples.shape))))
            assert new_vals_b.shape == new_mse_b.shape ==  torch.Size([batch_size])
            # Insert the calculated term into the tensor of all terms
            mask_bt = timesteps[:, None]==torch.arange(1000, device=timesteps.device)[None, :].float()
            vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
            mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt
            assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([batch_size, 1000])

        prior_bpd_b = self._prior_bpd(samples)
        total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b
        assert vals_bt_.shape == mse_bt_.shape == torch.Size([batch_size, 1000]) and \
                total_bpd_b.shape == prior_bpd_b.shape ==  torch.Size([batch_size])
        return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()


    # def validation_step(self, batch, batch_idx: int):
    #     pcs, labels = batch.pcs, batch.labels

    # learning rate warm-up
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # skip the first 500 steps
        if self.trainer.global_step < self.warmup_iters:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_iters)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate

    def configure_optimizers(self):
        # parameters = set_weight_decay(
        #     model=self,
        #     weight_decay=self.weight_decay,
        #     norm_weight_decay=self.norm_weight_decay,
        # )
        parameters = self.parameters()
        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "Adam":
            optimizer = torch.optim.Adam(
                parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.5, 0.999),
            )
        elif self.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError(self.optimizer_type)

        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.max_epochs
        # )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.998,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }


def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})

    return param_groups
