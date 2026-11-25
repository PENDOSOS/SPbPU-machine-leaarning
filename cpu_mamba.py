import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def selective_scan_cpu(u, delta, A, B, C, D, z, delta_bias):
    """Reference selective scan на PyTorch."""
    batch, dim, length = u.shape

    delta = F.softplus(delta + delta_bias.view(1, -1, 1))

    delta_a = torch.exp(
        torch.einsum("bdl,dn->bdln", delta, A)
    )
    delta_b_u = (
        torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    )

    state = torch.zeros(
        batch, dim, A.shape[-1],
        dtype=u.dtype,
        device=u.device,
    )

    outputs = []

    for i in range(length):
        state = (
            delta_a[:, :, i, :] * state
            + delta_b_u[:, :, i, :]
        )

        y = torch.einsum(
            "bdn,bn->bd",
            state,
            C[:, :, i],
        )
        outputs.append(y)

    y = torch.stack(outputs, dim=-1)
    y = y + u * D.view(1, -1, 1)
    y = y * F.silu(z)

    return y


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_divide_out=False,
        init_layer_scale=None,
    ):
        super().__init__()

        if bimamba_type not in ("none", "v2"):
            raise NotImplementedError(
                f"Unsupported bimamba_type: {bimamba_type}"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = (
            math.ceil(d_model / 16)
            if dt_rank == "auto"
            else dt_rank
        )

        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_divide_out = if_divide_out
        self.init_layer_scale = init_layer_scale
        self.use_fast_path = False

        kwargs = {"device": device, "dtype": dtype}

        if init_layer_scale is not None:
            self.gamma = nn.Parameter(
                init_layer_scale * torch.ones(d_model, **kwargs)
            )

        self.in_proj = nn.Linear(
            d_model, self.d_inner * 2,
            bias=bias, **kwargs
        )

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
            **kwargs,
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * d_state,
            bias=False,
            **kwargs,
        )

        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner,
            bias=True, **kwargs
        )

        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(
                    1, d_state + 1,
                    dtype=torch.float32,
                    device=device,
                ).repeat(self.d_inner, 1)
            )
        )

        self.D = nn.Parameter(
            torch.ones(self.d_inner, device=device)
        )

        if bimamba_type == "v2":
            self.A_b_log = nn.Parameter(self.A_log.detach().clone())

            self.conv1d_b = nn.Conv1d(
                self.d_inner,
                self.d_inner,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                bias=conv_bias,
                **kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner,
                self.dt_rank + 2 * d_state,
                bias=False,
                **kwargs,
            )

            self.dt_proj_b = nn.Linear(
                self.dt_rank, self.d_inner,
                bias=True, **kwargs
            )

            self.D_b = nn.Parameter(
                torch.ones(self.d_inner, device=device)
            )

        self.out_proj = nn.Linear(
            self.d_inner, d_model,
            bias=bias, **kwargs
        )

    def _scan(self, xz, backward=False):
        x, z = xz.chunk(2, dim=1)
        length = x.shape[-1]

        if backward:
            conv = self.conv1d_b
            x_proj = self.x_proj_b
            dt_proj = self.dt_proj_b
            A = -torch.exp(self.A_b_log.float())
            D = self.D_b.float()
        else:
            conv = self.conv1d
            x_proj = self.x_proj
            dt_proj = self.dt_proj
            A = -torch.exp(self.A_log.float())
            D = self.D.float()

        x = F.silu(conv(x)[..., :length])

        x_dbl = x_proj(
            rearrange(x, "b d l -> (b l) d")
        )

        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        dt = dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=length)

        B = rearrange(B, "(b l) n -> b n l", l=length)
        C = rearrange(C, "(b l) n -> b n l", l=length)

        return selective_scan_cpu(
            x, dt, A, B, C, D, z,
            dt_proj.bias.float(),
        )

    def forward(self, hidden_states, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError(
                "Inference cache is not implemented"
            )

        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")

        out = self._scan(xz)

        if self.bimamba_type == "v2":
            out_b = self._scan(
                xz.flip([-1]),
                backward=True,
            ).flip([-1])

            out = out + out_b

            if self.if_divide_out:
                out = out / 2

        out = rearrange(out, "b d l -> b l d")
        out = self.out_proj(out)

        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out