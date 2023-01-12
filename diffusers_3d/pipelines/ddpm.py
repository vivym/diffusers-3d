from typing import Optional

import torch
from torch import nn

from .pipeline import DiffusionPipeline


class DDPMPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()

        self.model = model
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        generator: Optional[torch.Generator] = None,
    ):
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
        """
        device = self.device
        assert generator.device == device

        if isinstance(self.model.sample_size, int):
            sample_shape = (
                batch_size, self.model.in_channels, self.model.sample_size, self.model.sample_size
            )
        else:
            sample_shape = (batch_size, self.model.in_channels, *self.model.sample_size)

        if device.type == "mps":
            # randn does not work reproducibly on mps
            sample = torch.randn(sample_shape, generator=generator)
            sample = sample.to(device)
        else:
            sample = torch.randn(sample_shape, generator=generator, device=device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps:
            # 1. predict noise model_output
            model_output = self.model(sample, t).sample

            # 2. compute previous image: x_t -> x_t-1
            sample = self.scheduler.step(
                model_output, t, sample, generator=generator
            ).prev_sample

        sample = (sample / 2 + 0.5).clamp(0, 1)

        return sample
