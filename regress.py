import os
import sys
import math
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from accelerate import Accelerator
from torch.utils.data import TensorDataset, DataLoader
from diffusers import UNet2DConditionModel

sys.path.append(os.path.dirname(__file__))
from compatible_pipelines import CompatibleLatentConsistencyModelPipeline
from ipattn import (
    MonkeyIPAttnProcessor,
    insert_monkey,
    reset_monkey,
    set_ip_adapter_scale_monkey,
)
from experiment_helpers.gpu_details import print_details
from experiment_helpers.argprint import print_args

_IP_TEST_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTwYThzD3l5DKWB4bRE842qIToTQIdkP5J1ZFEkG5icMQ&s"

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default=None, help="local image path; defaults to ip_test URL")
parser.add_argument("--model_layer", type=str, default="mid_block.attentions.0")
parser.add_argument("--steps", type=int, default=20, help="number of IP adapter scale steps")
parser.add_argument("--num_inference_steps", type=int, default=4)
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--prompt", type=str, default="person walking")
parser.add_argument("--negative_prompt", type=str, default="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality")
parser.add_argument("--token", type=int, default=1, help="which IP adapter token index to use for mask")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--mask_steps", nargs="*", type=int, default=None, help="denoising step indices to sum mask over; defaults to middle half")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--mlp_hidden", type=int, default=512)
parser.add_argument("--save_path", type=str, default="regress_mlp.pt")
parser.add_argument("--mixed_precision", type=str, default="no")
parser.add_argument("--test_frac", type=float, default=0.2, help="fraction of data held out for test")


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_mask(processor_kv: list, step: int, token: int, threshold: float) -> torch.Tensor:
    avg = processor_kv[step].mean(dim=1).squeeze(0)         # [seq_len, n_ip_tokens]
    latent_dim = int(math.sqrt(avg.size()[0]))
    avg = avg.view([latent_dim, latent_dim, -1])[:, :, token]  # [latent_dim, latent_dim]
    avg_min, avg_max = avg.min(), avg.max()
    x_norm = (avg - avg_min) / (avg_max - avg_min + 1e-8)
    x_norm[x_norm < threshold] = 0.0
    return x_norm


def main(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    if args.image_path:
        src_image = Image.open(args.image_path).convert("RGB")
    else:
        from diffusers.utils import load_image
        src_image = load_image(_IP_TEST_URL)

    pipe = CompatibleLatentConsistencyModelPipeline.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    setattr(pipe, "safety_checker", None)

    unet: UNet2DConditionModel = pipe.unet

    # Hook the target transformer block to cache its output features
    cached = {}

    def hook(module, input, output):
        t = output[0] if isinstance(output, tuple) else output
        cached["features"] = t.detach().float()  # [B, seq_len, hidden_dim]

    target = next((m for n, m in unet.named_modules() if n == args.model_layer), None)
    assert target is not None, f"Layer {args.model_layer!r} not found in UNet"
    target.register_forward_hook(hook)

    # Monkey-patch IP adapter attention processors to capture kv_ip
    set_ip_adapter_scale_monkey(pipe, 1.0)
    insert_monkey(pipe)

    # Resolve which monkey processors belong to args.model_layer
    layer_processors: list[MonkeyIPAttnProcessor] = [
        p for n, p in pipe.unet.attn_processors.items()
        if n.startswith(args.model_layer) and isinstance(p, MonkeyIPAttnProcessor)
    ]
    assert layer_processors, f"No MonkeyIPAttnProcessor found under {args.model_layer!r}"

    all_features: list[torch.Tensor] = []
    all_weights: list[float] = []

    with torch.no_grad():
        for step_idx in range(1, args.steps + 1):
            scale = step_idx / args.steps
            reset_monkey(pipe)
            set_ip_adapter_scale_monkey(pipe, scale)

            gen = torch.Generator(device=device).manual_seed(args.seed)
            pipe(
                prompt=args.prompt,
                ip_adapter_image=src_image,
                negative_prompt=args.negative_prompt,
                generator=gen,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=1.0,
                height=args.size,
                width=args.size,
            )

            # Determine which denoising steps to average mask over
            n_steps = len(layer_processors[0].kv_ip)
            if args.mask_steps is not None:
                mask_steps = [s for s in args.mask_steps if s < n_steps]
            else:
                quarter = n_steps // 4
                mask_steps = list(range(quarter, n_steps - quarter)) or list(range(n_steps))

            # Sum spatial IP attention masks across selected processors and steps
            mask: torch.Tensor | None = None
            for proc in layer_processors:
                for s in mask_steps:
                    if s < len(proc.kv_ip):
                        m = get_mask(proc.kv_ip, s, args.token, args.threshold)
                        mask = m if mask is None else mask + m

            if mask is None:
                continue

            

            # UNet features from the hook (last denoising step)
            unet_feats = cached["features"]
            print(unet_feats.size())
            #unet_feats=F.interpolate(unet_feats,mask.size()[-2:])
            mask=F.interpolate(mask.unsqueeze(0).unsqueeze(0),unet_feats.size()[-2:] )[0,0]
            
            print(mask.size())

            mask=torch.unbind(mask,0)
            unet_feats=torch.unbind(unet_feats,0)

            all_features.extend(unet_feats)
            all_weights.extend(mask * scale)

    assert all_features, "No training pairs collected — check layer name and processor setup"

    X = torch.stack(all_features)                                   # [N, hidden_dim]
    y = torch.tensor(all_weights, dtype=torch.float32)             # [N]

    n_test = max(1, int(len(X) * args.test_frac))
    n_train = len(X) - n_test
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    hidden_dim = X.shape[1]
    mlp = MLP(hidden_dim, args.mlp_hidden).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(X_train.to(device), y_train.to(device)),
        batch_size=min(args.batch_size, n_train),
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test.to(device), y_test.to(device)),
        batch_size=min(args.batch_size, n_test),
        shuffle=False,
    )

    mlp.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = F.mse_loss(mlp(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"epoch {epoch}: loss={total_loss / len(loader):.6f}")

    mlp.eval()
    test_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            test_loss += F.mse_loss(mlp(xb), yb).item()
    print(f"test loss={test_loss / len(test_loader):.6f}")

    torch.save(mlp.state_dict(), args.save_path)
    print(f"saved MLP to {args.save_path}")


if __name__ == "__main__":
    print_details()
    start = time.time()
    args = parser.parse_args()
    print_args(parser)
    main(args)
    end = time.time()
    seconds = end - start
    print(f"done in {seconds:.1f}s = {seconds/3600:.2f}h")
