import numpy as np
import torch
from tqdm import tqdm

from diffusers_3d.pl_modules.pvcnn import PVCNN
from diffusers_3d.models.pvcnn import pvcnn_debug
from diffusers_3d.structures.points import PointTensor


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
        for t in tqdm(reversed(range(0, self.num_timesteps if not keep_running else len(self.betas)))):
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


def main():
    device = torch.device("cuda")

    # model = PVCNN.load_from_checkpoint(
    #     "wandb/lightning_logs/version_20/checkpoints/epoch_000099.ckpt"
    # )
    model = PVCNN()
    model.model = pvcnn_debug()
    # print(model)

    state_dict = torch.load("../PVD/PVD/output/train_generation_pl_no_noise_init/2023-01-15-18-10-08/epoch_99.pth", map_location="cpu")
    print("state_dict", list(state_dict.keys()))
    new_state_dict = {}
    for k, v in state_dict["model_state"].items():
        # print(k, v.shape)
        k = k[len("model.module."):]
        # if k.startswith("embedf"):
        #     k = k.replace("embedf", "time_embed_proj")
        # elif k.startswith("classifier"):
        #     k = k.replace("classifier", "out_proj")
        # elif k.startswith("global_att"):
        #     k = k.replace("global_att", "global_attn")
        #     k = k.replace("q.", "q_proj.")
        #     k = k.replace("k.", "k_proj.")
        #     k = k.replace("v.", "v_proj.")
        #     k = k.replace("out.", "out_proj.")
        # elif k.startswith("sa_layers"):
        #     k = k.replace("sa_layers", "down_blocks")
        #     k = k.replace("point_features", "point_layers")
        #     k = k.replace("q.", "q_proj.")
        #     k = k.replace("k.", "k_proj.")
        #     k = k.replace("v.", "v_proj.")
        #     k = k.replace("out.", "out_proj.")
        #     if not k.split(".")[2].isnumeric():
        #         k = ".".join(k.split(".")[:2]) + f".0." + ".".join(k.split(".")[2:])
        # elif k.startswith("fp_layers"):
        #     k = k.replace("fp_layers", "up_blocks")
        #     k = k.replace("point_features", "point_layers")
        #     k = k.replace("q.", "q_proj.")
        #     k = k.replace("k.", "k_proj.")
        #     k = k.replace("v.", "v_proj.")
        #     k = k.replace("out.", "out_proj.")
        #     # fp_layers.0.0.mlp.layers.0.weight -> up_blocks.0.fp_module.mlp.layers.0.weight
        #     # fp_layers.0.1.voxel_layers.0.weight -> up_blocks.0.convs.0.voxel_layers.0.weight
        #     layer_idx = int(k.split(".")[2])
        #     if layer_idx == 0:
        #         k = ".".join(k.split(".")[:2]) + f".fp_module." + ".".join(k.split(".")[3:])
        #     else:
        #         k = ".".join(k.split(".")[:2]) + f".convs.{layer_idx - 1}." + ".".join(k.split(".")[3:])

        # print(k, v.shape)
        new_state_dict[k] = v

    model.model.load_state_dict(new_state_dict)

    model.model.eval()
    model.to(device)

    sample_shape = (16, 2048, 3)

    sample = torch.randn(sample_shape, device=device)

    # set step values
    model.noise_scheduler.set_timesteps(1000)

    betas = np.linspace(0.0001, 0.02, 1000)
    diffusion = GaussianDiffusion(betas, "mse", "eps", "fixedsmall")

    def _denoise(data, t):
        B, D,N= data.shape
        # print("data", data.shape)
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        points = PointTensor(
            coords=data[:, :3, :].permute(0, 2, 1),
            features=data,
            is_channel_last=False,
            timesteps=t,
        )

        out = model.model(points)
        # out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def gen_samples(shape, device, noise_fn=torch.randn,
                    clip_denoised=True,
                    keep_running=False):
        return diffusion.p_sample_loop(_denoise, shape=shape, device=device, noise_fn=noise_fn,
                                            clip_denoised=clip_denoised,
                                            keep_running=keep_running)

    with torch.no_grad():
        sample = gen_samples((16, 3, 2048), device, clip_denoised=False)

    # with torch.no_grad():
    #     for t in tqdm(model.noise_scheduler.timesteps):
    #         # 1. predict noise model_output
    #         model_output = model(sample, t)

    #         # 2. compute previous image: x_t -> x_t-1
    #         sample = model.noise_scheduler.step(
    #             model_output, t, sample
    #         ).prev_sample

    print("sample", sample.shape)
    torch.save(sample, "generated.pth")


if __name__ == "__main__":
    main()
