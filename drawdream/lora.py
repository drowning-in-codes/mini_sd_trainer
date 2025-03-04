import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save

    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False


class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features,out_features)}"
            )
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        #  init ΔW = A*B
        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return (
            self.linear(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        assert diag.shape == (self.r,)
        self.selector = nn.Linear(self.r, self.r, bias=False)
        self.selector.weight.data = (
            torch.diag(diag)
            .to(self.lora_up.weight.device)
            .to(self.lora_up.weight.dtype)
        )


class LoraInjectedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        r: int = 4,
        dropout_p: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        if r > min(in_channels, out_channels):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}"
            )
        self.r = r
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.lora_down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=r,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv2d(
            in_channels=r,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return (
            self.conv(input)
            + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
            * self.scale
        )

    def realize_as_lora(self):
        return self.lora_up.weight.data * self.scale, self.lora_down.weight.data

    def set_selector_from_diag(self, diag: torch.Tensor):
        assert diag.shape == (self.r,)
        self.selector = nn.Conv2d(
            in_channels=self.r,
            out_channels=self.r,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.selector.weight.data = torch.diag(diag)
        self.selector.weight.data = self.selector.weight.data.to(
            self.lora_up.weight.device
        ).to(self.lora_up.weight.dtype)


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}

UNET_EXTENDED_TARGET_REPLACE = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}

TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}

TEXT_ENCODER_EXTENDED_TARGET_REPLACE = {"CLIPAttention"}

DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE

EMBED_FLAG = "<embed>"


def _find_modules_v2(
    model,
    ancestor_class: Optional[set[str]] = None,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [
        LoraInjectedLinear,
        LoraInjectedConv2d,
    ],
):
    if ancestor_class is not None:
        ancestors: list[nn.Module] = (
            module
            for module in model.modules()
            if module.__class__.__name__ in ancestor_class
        )
    else:
        ancestors: list[nn.Module] = [module for module in model.modules()]
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                if exclude_children_of and any(
                    [isinstance(parent, _class) for _class in exclude_children_of]
                ):
                    continue
                yield parent, name, module


_find_modules = _find_modules_v2


def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 4,
    loras=None,
    verbose: bool = False,
    dropout_p: float = 0.0,
    scale: float = 1.0,
):
    require_grad_params = []
    names = []
    if loras != None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear]
    ):
        weight = _child_module.weight
        bias = _child_module.bias
        if verbose:
            print("LoRA Injection: injecting lora into ", name)
            print("LoRA Injection: weight shape", weight.shape)
        _tmp = LoraInjectedLinear(
            in_features=_child_module.in_features,
            out_features=_child_module.out_features,
            bias=_child_module.bias is not None,
            r=r,
            dropout_p=dropout_p,
            scale=scale,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias
            nn.Module

        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)
        _module._modules[name].lora_up.weight.requires_grad_()
        _module._modules[name].lora_down.weight.requires_grad_()
        names.append(name)
    return require_grad_params, names


def inject_trainable_lora_extended(
    model: nn.Module,
    target_replace_module: set[str] = UNET_EXTENDED_TARGET_REPLACE,
    r: int = 4,
    loras=None,
):
    require_grad_params = []
    names = []

    if loras is not None:
        loras = torch.load(loras)

    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, nn.Conv2d]
    ):
        if _child_module.__class__ == nn.Linear:
            weight = _child_module.weight
            bias = _child_module.bias

            _tmp = LoraInjectedLinear(
                in_features=_child_module.in_features,
                out_features=_child_module.out_features,
                bias=_child_module.bias is not None,
                r=r,
            )
            _tmp.linear.weight = weight
            if bias is not None:
                _tmp.linear.bias = bias
        elif _child_module.__class__ == nn.Conv2d:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = LoraInjectedConv2d(
                in_channels=_child_module.in_channels,
                out_channels=_child_module.out_chanels,
                kernel_size=_child_module.kernel_size,
                stride=_child_module.stride,
                padding=_child_module.padding,
                dilation=_child_module.dilation,
                groups=_child_module.groups,
                bias=_child_module.bias is not None,
                r=r,
            )
            _tmp.conv.weight = weight
            if bias is not None:
                _tmp.conv.bias = bias
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp
        require_grad_params.append(_tmp.lora_up.parameters())
        require_grad_params.append(_tmp.lora_down.parameters())
        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)
        _module._modules[name].lora_up.weight.requires_grad_()
        _module._modules[name].lora_down.weight.requires_grad_()
        names.append(name)
    return require_grad_params, names





def extract_lora_ups_down(model, target_replace_module=DEFAULT_TARGET_REPLACE):
    loras = []

    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        up = _child_module.lora_up
        down = _child_module.lora_down
        loras.append((up, down))

    if len(loras) == 0:
        raise ValueError("No lora injected")
    return loras


def extract_lora_as_tensor(
    model, target_replace_module=DEFAULT_TARGET_REPLACE, as_fp16=True
):
    loras = []

    for _m, _n, _child_module in _find_modules(
        model,
        target_replace_module,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        _up, _down = _child_module.realize_as_lora()
        if as_fp16:
            _up = _up.to(torch.float16)
            _down = _down.to(torch.float16)
        loras.append((_up, _down))

    if len(loras) == 0:
        raise ValueError("No lora injected")
    return loras


def save_lora_weight(
    model, path="./lora.pt", target_replace_module=DEFAULT_TARGET_REPLACE
):
    weights = []
    for _up, _down in extract_lora_ups_down(model, target_replace_module):
        weights.append(_up.weight.to("cpu").to(torch.float16))
        weights.append(_down.weight.to("cpu").to(torch.float16))
    torch.save(weights, path)


def save_lora_as_json(model, path="./lora.json"):
    weights = []
    for _up, _down in extract_lora_ups_down(model):
        weights.append(_up.weight.detach().cpu().numpy().tolist())
        weights.append(_down.weight.detach().cpu().numpy().tolist())
    with open(path, "w") as f:
        json.dump(weights, f)


def save_safeloras_with_embeds(
    modelmap: dict[str, tuple[nn.Module, set[str]]] = {},
    embeds: dict[str, torch.Tensor] = {},
    outpath="./lora.safetensor",
):
    weights = {}
    metadata = {}
    for name, (model, target_replace_module) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))

        for i, (_up, _down) in enumerate(
            extract_lora_as_tensor(model, target_replace_module)
        ):
            rank = _down.shape[0]

            metadata[f"{name}:{i}:rank"] = str(rank)
            weights[f"{name}:{i}:up"] = _up
            weights[f"{name}:{i}:down"] = _down

    for token, tensor in embeds.items():
        metadata[token] = EMBED_FLAG
        weights[token] = tensor

    print(f"Saving weights to {outpath}")
    safe_save(weights, outpath, metadata)


def save_safeloras(
    modelmap: dict[str, tuple[nn.Module, set[str]]] = {}, outpath="./lora.safetensors"
):
    return save_safeloras_with_embeds(modelmap=modelmap, outpath=outpath)


def convert_loras_to_safeloras_with_embeds(
    modelmap: dict[str, tuple[str, set[str], int]] = {},
    embeds: dict[str, torch.Tensor] = {},
    outpath="./lora.safetensors",
):
    weights = {}
    metadata = {}
    for name, (path, target_replace_module, r) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))
        lora = torch.load(path)
        for i, weight in enumerate(lora):
            is_up = i % 2 == 0
            i = i // 2
            if is_up:
                metadata[f"{name}:{i}:rank"] = str(r)
                weights[f"{name}:{i}:up"] = weight
            else:
                weights[f"{name}:{i}:down"] = weight

    for token, tensor in embeds.items():
        weights[token] = tensor
        metadata[token] = EMBED_FLAG

    print(f"Saving weights to {outpath}")
    safe_save(weights, outpath, metadata)


def convert_loras_to_safeloras(
    modelmap: dict[str, tuple[str, set[str], int]] = {}, outpath="./lora.safetensors"
):
    convert_loras_to_safeloras_with_embeds(modelmap=modelmap, outpath=outpath)


def parse_safeloras(safeloras):
    loras = {}
    metadata = safeloras.metadata()
    get_name = lambda k: k.splits(":")[0]

    keys = list(safeloras.keys())
    keys.sort(key=get_name)
    for name, module_keys in groupby(keys, get_name):
        info = metadata.get(name)
        if not info:
            raise ValueError(
                f"Tensor {name} has no metadata - is this a Lora safetensor?"
            )
        # Skip Textual Inversion embeds
        if info == EMBED_FLAG:
            continue

        target = json.loads(info)
        module_keys = list(module_keys)
        ranks = [4] * (len(module_keys) // 2)
        weights = [None] * len(module_keys)
        for key in module_keys:
            _, idx, direction = key.split(":")
            idx = int(idx)

            ranks[idx] = int(metadata[f"{name}:{idx}:rank"])

            idx = idx * 2 + (1 if direction == "down" else 0)
            weights[idx] = nn.Parameter(safeloras.get_tensor(key))
        loras[name] = (weights, ranks, target)
    return loras


def parse_safeloras_embeds(safeloras):
    embeds = {}
    metadata = safeloras.metadata()

    for key in safeloras.keys():
        meta = metadata.get(key)
        if not meta or meta != EMBED_FLAG:
            continue
        embeds[key] = safeloras.get_tensor(key)
    return embeds


def load_safeloras(path, device="cpu"):
    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras(safeloras)


def load_safeloras_embeds(path, device="cpu"):
    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras_embeds(safeloras)


def load_safeloras_both(path, device="cpu"):
    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras(safeloras), parse_safeloras_embeds(safeloras)


def collapse_lora(model, alpha=1.0):
    for _module, name, _child_module in _find_modules(
        model,
        UNET_EXTENDED_TARGET_REPLACE | TEXT_ENCODER_EXTENDED_TARGET_REPLACE,
        search_class=[LoraInjectedLinear, LoraInjectedConv2d],
    ):
        if isinstance(_child_module, LoraInjectedLinear):
            print("Collapsing Lin Lora in ", name)
            _child_module.linear.weight = nn.Parameter(
                _child_module.linear.weight.data
                + alpha
                * (
                    _child_module.lora_up.weight.data
                    @ _child_module.lora_down.weight.data
                )
                .type(_child_module.linear.weight.dtype)
                .to(_child_module.linear.weight.device)
            )
        else:
            print("Collapsing Conv Lora in ", name)
            _child_module.conv.weight = nn.Parameter(
                _child_module.conv.weight
                + alpha
                * (
                    _child_module.lora_up.weight.data
                    @ _child_module.lora_down.weight.data
                )
                .type(_child_module.lora_up.weight.dtype)
                .to(_child_module.lora_up.weight.device)
            )


def monkeypatch_or_replace_lora(
    model, loras, target_replace_module=DEFAULT_TARGET_REPLACE, r: int | list[int] = 4
):
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, [nn.Linear, LoraInjectedLinear]
    ):
        _source = (
            _child_module.linear
            if isinstance(_child_module, LoraInjectedLinear)
            else _child_module
        )
        weight = _source.weight
        bias = _source.bias
        _tmp = LoraInjectedLinear(
            _source.in_features,
            _source.out_features,
            _source.bias is not None,
            r=r.pop(0) if isinstance(r, list) else r,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias
        _module._modules[name] = _tmp
        up_weight = loras.pop(0)
        down_weight = loras.pop(0)
        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.type)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.type)
        )
        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_lora_extended(
    model, loras, target_replace_model=DEFAULT_TARGET_REPLACE, r: int | list[int] = 4
):
    for _module, name, _child_module in _find_modules(
        model,
        target_replace_model,
        search_class=[nn.Linear, LoraInjectedLinear, nn.Conv2d, LoraInjectedConv2d],
    ):
        if (
            _child_module.__class__ == nn.Linear
        ) or _child_module.__class__ == LoraInjectedLinear:
            if (len(loras[0].shape)) != 2:
                continue

            _source = (
                _child_module.linear
                if isinstance(_child_module, LoraInjectedLinear)
                else _child_module
            )
            weight = _source.weight
            bias = _source.bias
            _tmp = LoraInjectedLinear(
                _source.in_features,
                _source.out_features,
                _source.bias is not None,
                r=r.pop(0) if isinstance(r, list) else r,
            )
            _tmp.linear.weight = weight
            if bias is not None:
                _tmp.linear.bias = bias
        elif (
            _child_module.__class__ == nn.Conv2d
            or _child_module.__class__ == LoraInjectedConv2d
        ):
            if (len(loras[0].shape)) != 2:
                continue

            _source = (
                _child_module.conv
                if isinstance(_child_module, LoraInjectedConv2d)
                else _child_module
            )
            weight = _source.weight
            bias = _source.bias
            _tmp = LoraInjectedConv2d(
                _source.in_channels,
                _source.out_channels,
                _source.stride,
                _source.padding,
                _source.dilation,
                _source.groups,
                _source.bias is not None,
                r=r.pop(0) if isinstance(r, list) else r,
            )
            _tmp.conv.weight = weight
            if bias is not None:
                _tmp.conv.bias = bias
        _module._modules[name] = _tmp
        up_weight = loras.pop(0)
        down_weight = loras.pop(0)
        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype)
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype)
        )
        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_safeloras(models, safeloras):
    loras = parse_safeloras(safeloras)
    for name, (lora, ranks, target) in loras.items():
        model = getattr(models, name, None)
        if not model:
            print(f"No model provided for {name}, contained in Lora")
            continue
        monkeypatch_or_replace_lora_extended(model, lora, target, ranks)


def monkeypatch_remove_lora(model):
    for _module, name, _child_module in _find_modules(
        model,
        search_class=[LoraInjectedConv2d, LoraInjectedLinear],
    ):
        if isinstance(_child_module, LoraInjectedLinear):
            _source = _child_module.linear
            weight, bias = _source.weight, _source.bias
            _tmp = nn.Linear(
                _source.in_features, _source.out_features, bias=_source.bias is not None
            )
            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias
        else:
            _source = _child_module.conv

            weight, bias = _source.weight, _source.bias
            _tmp = nn.Conv2d(
                _source.in_channels,
                _source.out_channels,
                _source.kernel_size,
                _source.stride,
                _source.padding,
                _source.dilation,
                _source.groups,
                bias=_source.bias is not None,
            )
            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias
        _module._modules[name] = _tmp.to(_source.weight.device)


def monkeypatch_add_lora(
    model, loras, target_replace_module=DEFAULT_TARGET_REPLACE, alpha=1.0, beta=1.0
):
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[LoraInjectedLinear]
    ):
        weight = _child_module.linear.weight
        up_weight = loras.pop(0)
        down_weight = loras.pop(0)
        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name].lora_up.weight.type(weight.dtype).to(weight.device)
            * beta
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name]
            .lora_down.weight.type(weight.dtype)
            .to(weight.device)
            * beta
        )
        _module._modules[name].to(weight.device)


def tune_lora_scale(model, alpha: float = 1.0):
    for _module in model.modules():
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            _module.scale = alpha


def set_lora_diag(model, diag: torch.Tensor):
    for _module in model.modules():
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            _module.set_selector_from_diag(diag)


def _text_lora_path(path: str):
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[-1] + ["text_encoder", "pt"])


def _ti_lora_path(path: str):
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[-1] + ["ti", "pt"])


def apply_learned_embed_in_clip(
    learned_embeds,
    text_encoder,
    tokenizer,
    token: None | str | list[str] = None,
    idempotent=False,
):
    if isinstance(token, str):
        trained_token = [token]
    elif isinstance(token, list):
        assert len(token) == len(learned_embeds.keys())
        trained_token = token
    else:
        trained_token = learned_embeds.keys()

    for token in trained_token:
        embeds = learned_embeds[token]
        dtype = text_encoder.get_input_embeddings().weight.dtype
        num_added_tokens = tokenizer.add_token(token)

        i = 1
        if not idempotent:
            while num_added_tokens == 0:
                print(f"The tokenizer already contains the token {token}")
                token = f"{token[:-1]}-{i}>"
                print(f"Attempting to add new token {token}")
                num_added_tokens = tokenizer.add_token(token)
                i += 1
        else:
            if num_added_tokens == 0:
                print(f"The tokenizer already contains the token {token}")
                print("Replace this token embedding")
        text_encoder.resize_token_embeddings(len(tokenizer))

        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight[token_id] = embeds.to(dtype)
    return token


def load_learned_embed_in_clip(
    learned_embeds_path,
    text_encoder,
    tokenizer,
    token: None | str | list[str] = None,
    idempotent=False,
):

    learned_embeds = torch.load(learned_embeds_path)
    return apply_learned_embed_in_clip(
        learned_embeds, text_encoder, tokenizer, token, idempotent
    )


def patch_pipe(
    pipe,
    maybe_unet_path,
    token: None | str = None,
    r: int = 4,
    patch_unet=True,
    patch_text=True,
    patch_ti=True,
    idempotent_token=True,
    unet_target_replace_module=DEFAULT_TARGET_REPLACE,
    text_target_replace_module=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
):
    if maybe_unet_path.endswith(".pt"):
        if maybe_unet_path.endswith(".ti.pt"):
            unet_path = maybe_unet_path[:-6] + ".pt"
        elif maybe_unet_path.endswith(".text_encoder.pt"):
            unet_path = maybe_unet_path[:-16] + ".pt"
        else:
            unet_path = maybe_unet_path
        ti_path = _ti_lora_path(unet_path)
        text_path = _text_lora_path(unet_path)

        if patch_unet:
            print("LoRA: Patching Unet")
            monkeypatch_or_replace_lora(
                pipe.unet,
                torch.load(unet_path),
                r=r,
                target_replace_module=unet_target_replace_module,
            )

        if patch_text:
            print("LoRA: Patching text encoder")
            monkeypatch_or_replace_lora(
                pipe.text_encoder,
                torch.load(text_path),
                target_replace_module=text_target_replace_module,
                r=r,
            )
        if patch_ti:
            print("LoRA: Patching token input")
            token = load_learned_embed_in_clip(
                ti_path, pipe.text_encoder, pipe.tokenizer, token, idempotent_token
            )
    elif maybe_unet_path.endswith(".safetensors"):
        safeloras = safe_open(maybe_unet_path, framework="pt", device="cpu")
        monkeypatch_or_replace_safeloras(pipe, safeloras)
        tok_dict = parse_safeloras_embeds(safeloras)

        if patch_ti:
            apply_learned_embed_in_clip(
                tok_dict, pipe.text_encoder, pipe.tokenizer, token, idempotent_token
            )
        return tok_dict


@torch.no_grad()
def inspect_lora(model):
    moved = {}
    for name, _module in model.named_modules():
        if _module.__class__.__name__ in ["LoraInjectedLinear", "LoraInjectedConv2d"]:
            ups = _module.lora_up.weight.data.clone()
            downs = _module.lora_down.weight.data.clone()

            wght = ups.flatten(1) @ downs.flatten(1)
            dist = wght.flatten().abs().mean().item()
            if name in moved:
                moved[name].append(dist)
            else:
                moved[name] = [dist]
    return moved


def save_all(
    unet,
    text_encoder,
    save_path,
    placeholder_token_ids=None,
    placeholder_tokens=None,
    save_lora=None,
    save_ti=None,
    target_replace_module_text=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    target_replce_module_unet=DEFAULT_TARGET_REPLACE,
    safe_form=True,
):
    if not safe_form:
        if save_ti:
            ti_path = _ti_lora_path(save_path)
            learned_embeds_dict = {}
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                print(f"Current Learned Embeddings for {tok}:, id {tok_id}")
                learned_embeds_dict[tok] = learned_embeds.detach().cpu()
            torch.save(learned_embeds_dict, ti_path)
            print("Ti saved to ", ti_path)
        if save_lora:
            save_lora_weight(
                unet, save_path, target_replace_module=target_replce_module_unet
            )
            print("Unet saved to ", save_path)
    else:
        assert save_path.endswith(
            ".safetensors"
        ), f"Save path: {save_path} should end with .safetensors"

        loras = {}
        embeds = {}

        if save_lora:
            loras["unet"] = (unet, target_replce_module_unet)
            loras["text_encoder"] = (text_encoder, target_replace_module_text)
        if save_ti:
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                print(f"Current Learned Embeddings for {tok}: id {tok_id}")
                embeds[tok] = learned_embeds.detach().cpu()

        save_safeloras_with_embeds(loras, embeds, save_path)


if __name__ == "__main__":
    submodule = nn.Sequential(
        nn.Conv2d(
            10,
            10,
            3,
        ),
        nn.Conv2d(
            10,
            10,
            3,
        ),
    )
    net = nn.Sequential(submodule, nn.ReLU())
    # modules_found = list(_find_modules_v2(net))
    # print(modules_found)
    print(list(net.modules()) == set([module for module in net.modules()]))
    print(list(net.modules()))
    components = []
    for module in net:
        print(module)
