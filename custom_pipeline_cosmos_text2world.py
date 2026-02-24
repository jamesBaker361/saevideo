from diffusers import CosmosTextToWorldPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLCosmos, CosmosTransformer3DModel
from diffusers.schedulers import EDMEulerScheduler
from diffusers.utils import is_cosmos_guardrail_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.cosmos.pipeline_output import CosmosPipelineOutput
import inspect
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import T5EncoderModel, T5TokenizerFast

class CustomCosmosTextToWorldPipeline(CosmosTextToWorldPipeline):
    def __init__(
            self,
            text_encoder: T5EncoderModel,
            tokenizer: T5TokenizerFast,
            transformer: CosmosTransformer3DModel,
            vae: AutoencoderKLCosmos,
            scheduler: EDMEulerScheduler,
            safety_checker= None,
        ):
            super(DiffusionPipeline,self).__init__()


            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer,
                scheduler=scheduler,
                safety_checker=None,
            )

            self.vae_scale_factor_temporal = (
                self.vae.config.temporal_compression_ratio if getattr(self, "vae", None) else 8
            )
            self.vae_scale_factor_spatial = self.vae.config.spatial_compression_ratio if getattr(self, "vae", None) else 8
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)