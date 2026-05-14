# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import random
import textwrap
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState, logging
from accelerate.utils import tqdm
from datasets import Dataset, IterableDataset
from torch import autocast
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.integrations import (
    is_comet_available,
    is_mlflow_available,
    is_wandb_available,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_liger_kernel_available, is_peft_available

from ..data_utils import maybe_apply_chat_template, maybe_extract_prompt
from ..models import create_reference_model, prepare_deepspeed
from ..models.utils import prepare_fsdp
from .base_trainer import BaseTrainer
from .callbacks import SyncRefModelCallback
from .dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
from .utils import (
    RunningMoments,
    cap_exp,
    create_model_from_path,
    disable_dropout_in_model,
    empty_cache,
    flush_left,
    flush_right,
    get_config_model_id,
    log_table_to_comet_experiment,
    pad,
    pad_to_length,
    peft_module_casting_to_bf16,
    selective_log_softmax,
)


if is_peft_available():
    from peft import (
        PeftConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss


if is_wandb_available():
    import wandb

if is_mlflow_available():
    import mlflow


logger = logging.get_logger(__name__)


def _get_mode(args: Any) -> str:
    """Return training baseline mode."""
    m = getattr(args, "mode", None)
    if m is None:
        # Backward compatible default: original behavior
        return "raco" if getattr(args, "raco", False) else "default"
    return str(m).strip().lower()


def _grads_dot(a: list[torch.Tensor | None], b: list[torch.Tensor | None]) -> torch.Tensor:
    """Dot product between two gradient lists (sum over parameters). Returns a scalar tensor on a's device."""
    out = None
    for ga, gb in zip(a, b, strict=False):
        if ga is None or gb is None:
            continue
        v = (ga * gb).sum()
        out = v if out is None else out + v
    if out is None:
        # Fallback: no grads present (keep device/dtype if possible)
        for g in a:
            if g is not None:
                return torch.zeros((), device=g.device, dtype=g.dtype)
        for g in b:
            if g is not None:
                return torch.zeros((), device=g.device, dtype=g.dtype)
        return torch.tensor(0.0)
    return out


def _grads_norm(a: list[torch.Tensor | None]) -> torch.Tensor:
    """L2 norm of a gradient list. Returns a scalar tensor."""
    return torch.sqrt(torch.clamp(_grads_dot(a, a), min=0.0))


def _safe_scalar_ratio(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Finite scalar ratio num / max(den, eps) for logging/debugging."""
    den_safe = torch.clamp(den.abs(), min=eps)
    return num / den_safe


def _safe_scalar_cosine(
    dot: torch.Tensor,
    norm_a: torch.Tensor,
    norm_b: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Finite cosine similarity dot / (||a|| ||b||), returning 0 when either norm is ~0."""
    denom = norm_a * norm_b
    if float(denom.detach().abs().item()) <= eps:
        return torch.zeros((), device=dot.device, dtype=dot.dtype)
    cos = dot / torch.clamp(denom.abs(), min=eps)
    return torch.clamp(cos, min=-1.0, max=1.0)


def _grads_scale(a: list[torch.Tensor | None], s: float) -> list[torch.Tensor | None]:
    if s == 1.0:
        return a
    out: list[torch.Tensor | None] = []
    for g in a:
        out.append(None if g is None else g * s)
    return out


def _grads_add(a: list[torch.Tensor | None], b: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
    out: list[torch.Tensor | None] = []
    for ga, gb in zip(a, b, strict=False):
        if ga is None and gb is None:
            out.append(None)
        elif ga is None:
            out.append(gb)
        elif gb is None:
            out.append(ga)
        else:
            out.append(ga + gb)
    return out


def _grads_affine(u1: list[torch.Tensor | None], u2: list[torch.Tensor | None], lam: float) -> list[torch.Tensor | None]:
    """Return lam*u1 + (1-lam)*u2."""
    out: list[torch.Tensor | None] = []
    for a, b in zip(u1, u2, strict=False):
        if a is None and b is None:
            out.append(None)
        elif a is None:
            out.append(b * (1.0 - lam))
        elif b is None:
            out.append(a * lam)
        else:
            out.append(a * lam + b * (1.0 - lam))
    return out


def _raco_solve_lambda_k2(
    b1: torch.Tensor,
    b2: torch.Tensor,
    H11: torch.Tensor,
    H12: torch.Tensor,
    H22: torch.Tensor,
    s: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[float, dict[str, float]]:
    """
    Closed-form K=2 solver described by user:
      minimize h(λ) = b2 + (b1-b2)λ + s*sqrt(Q(λ)), λ∈[0,1]
    where Q(λ)=q2 λ^2 + q1 λ + q0 with q2=H11+H22-2H12, q1=2(H12-H22), q0=H22.
    """
    # Move everything to float64 for numeric stability
    b1 = b1.double()
    b2 = b2.double()
    H11 = H11.double()
    H12 = H12.double()
    H22 = H22.double()
    s = s.double()

    delta = b1 - b2
    q2 = H11 + H22 - 2.0 * H12
    q1 = 2.0 * (H12 - H22)
    q0 = H22

    def Q(lam_t: torch.Tensor) -> torch.Tensor:
        return q2 * lam_t * lam_t + q1 * lam_t + q0

    def h(lam_t: torch.Tensor) -> torch.Tensor:
        return b2 + delta * lam_t + s * torch.sqrt(torch.clamp(Q(lam_t), min=0.0))

    # Quadratic for h'(λ)=0:
    # (δ^2 q2 - s^2 q2^2) λ^2 + (δ^2 q1 - s^2 q1 q2) λ + (δ^2 q0 - s^2 q1^2/4) = 0
    A = (delta * delta) * q2 - (s * s) * (q2 * q2)
    B = (delta * delta) * q1 - (s * s) * (q1 * q2)
    C = (delta * delta) * q0 - (s * s) * (q1 * q1) / 4.0

    cands: list[float] = [0.0, 1.0]

    # Solve quadratic (or linear) in float64
    if torch.abs(A) < eps:
        if torch.abs(B) >= eps:
            root = (-C / B).item()
            if 0.0 <= root <= 1.0:
                cands.append(float(root))
    else:
        disc = (B * B) - 4.0 * A * C
        if disc >= 0:
            sqrt_disc = torch.sqrt(disc)
            r1 = ((-B + sqrt_disc) / (2.0 * A)).item()
            r2 = ((-B - sqrt_disc) / (2.0 * A)).item()
            for r in (r1, r2):
                if 0.0 <= r <= 1.0:
                    cands.append(float(r))

    # Evaluate candidates
    best_lam = 0.0
    best_val = None
    for lam in cands:
        lam_t = torch.tensor(lam, dtype=torch.float64, device=b1.device)
        val = h(lam_t)
        if best_val is None or val < best_val:
            best_val = val
            best_lam = lam

    dbg = {
        "delta": float(delta.item()),
        "b1": float(b1.item()),
        "b2": float(b2.item()),
        "H11": float(H11.item()),
        "H12": float(H12.item()),
        "H22": float(H22.item()),
        "s": float(s.item()),
        "q2": float(q2.item()),
        "q1": float(q1.item()),
        "q0": float(q0.item()),
        "lambda": float(best_lam),
        "h": float(best_val.item()) if best_val is not None else float("nan"),
    }
    return float(best_lam), dbg


def _raco_real_roots_quadratic(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    eps: float = 1e-12,
) -> list[float]:
    """Return unique real roots of a*x^2 + b*x + c = 0."""
    a = a.double()
    b = b.double()
    c = c.double()

    roots: list[float] = []
    if torch.abs(a) < eps:
        if torch.abs(b) >= eps:
            roots.append(float((-c / b).item()))
    else:
        disc = (b * b) - 4.0 * a * c
        if disc >= -eps:
            disc = torch.clamp(disc, min=0.0)
            sqrt_disc = torch.sqrt(disc)
            roots.append(float(((-b + sqrt_disc) / (2.0 * a)).item()))
            roots.append(float(((-b - sqrt_disc) / (2.0 * a)).item()))

    uniq: list[float] = []
    for r in roots:
        if not any(abs(r - u) <= 1e-10 for u in uniq):
            uniq.append(r)
    return uniq


def _raco_eval_phi_k3(p: torch.Tensor, b: torch.Tensor, H: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    p = p.double()
    b = b.double()
    H = H.double()
    s = s.double()
    quad = torch.dot(p, H @ p)
    return torch.dot(b, p) + s * torch.sqrt(torch.clamp(quad, min=0.0))


def _raco_add_candidate_k3(
    candidates: list[tuple[float, torch.Tensor, dict[str, float]]],
    p: torch.Tensor,
    b: torch.Tensor,
    H: torch.Tensor,
    s: torch.Tensor,
    *,
    case_id: float,
    simplex_tol: float = 1e-8,
) -> None:
    """Validate and append a simplex candidate for the exact K=3 solver."""
    p = p.detach().clone().double()
    if p.shape != (3,) or not torch.isfinite(p).all():
        return

    p = torch.where((p < 0.0) & (p > -simplex_tol), torch.zeros_like(p), p)
    if torch.any(p < -simplex_tol):
        return

    total = float(p.sum().item())
    if abs(total) <= simplex_tol:
        return
    if abs(total - 1.0) > 10.0 * simplex_tol:
        return
    if abs(total - 1.0) > simplex_tol:
        p = p / total

    phi = _raco_eval_phi_k3(p, b, H, s)
    candidates.append(
        (
            float(phi.item()),
            p,
            {
                "case_id": float(case_id),
                "phi": float(phi.item()),
                "p1": float(p[0].item()),
                "p2": float(p[1].item()),
                "p3": float(p[2].item()),
            },
        )
    )


def _raco_zero_norm_interior_candidate_k3(
    H: torch.Tensor,
    *,
    pos_tol: float = 1e-8,
    rank_tol: float = 1e-10,
) -> torch.Tensor | None:
    """
    Find an interior simplex point p > 0 with H p = 0, if one exists.
    For K=3 this can be solved explicitly from the nullspace of H.
    """
    H = H.double()
    device = H.device
    dtype = H.dtype
    ones = torch.ones(3, device=device, dtype=dtype)

    evals, evecs = torch.linalg.eigh(H)
    eval_scale = max(1.0, float(torch.max(torch.abs(evals)).item()))
    tol = rank_tol * eval_scale
    null_mask = torch.abs(evals) <= tol
    null_dim = int(null_mask.sum().item())
    if null_dim == 0:
        return None

    if null_dim == 3:
        return torch.full((3,), 1.0 / 3.0, device=device, dtype=dtype)

    N = evecs[:, null_mask]
    a = N.transpose(0, 1) @ ones
    if float(torch.linalg.vector_norm(a).item()) <= tol:
        return None

    if null_dim == 1:
        coeff = float(a[0].item())
        if abs(coeff) <= tol:
            return None
        p = N[:, 0] / coeff
        if torch.all(p > pos_tol):
            return p
        return None

    if null_dim == 2:
        z0 = a / torch.dot(a, a)
        p0 = N @ z0
        if torch.all(p0 > pos_tol):
            return p0

        z_dir = torch.stack((-a[1], a[0]))
        d = N @ z_dir

        lower = -float("inf")
        upper = float("inf")
        for i in range(3):
            pi = float(p0[i].item())
            di = float(d[i].item())
            if abs(di) <= tol:
                if pi <= pos_tol:
                    return None
                continue
            bound = (pos_tol - pi) / di
            if di > 0:
                lower = max(lower, bound)
            else:
                upper = min(upper, bound)

        if not (lower < upper):
            return None
        if lower == -float("inf") and upper == float("inf"):
            beta = 0.0
        elif lower == -float("inf"):
            beta = upper - 1.0
        elif upper == float("inf"):
            beta = lower + 1.0
        else:
            beta = 0.5 * (lower + upper)
        p = p0 + (beta * d)
        if torch.all(p > pos_tol):
            return p
        return None

    return None


def _raco_solve_p_k3_exact(
    b: torch.Tensor,
    H: torch.Tensor,
    s: torch.Tensor,
    *,
    eps: float = 1e-12,
    simplex_tol: float = 1e-8,
    pos_tol: float = 1e-8,
    consistency_tol: float = 1e-7,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Exact K=3 solver for:
      minimize  Phi(p) = b^T p + s * sqrt(p^T H p)
      subject to p in Delta_3.

    Candidate set:
      - 3 vertices
      - 3 edge minimizers (via the existing K=2 closed-form solver)
      - interior positive-norm KKT candidates
      - interior zero-norm candidates (Hp = 0, p > 0, 1^T p = 1)
    """
    b = b.double()
    H = H.double()
    s = s.double()
    device = b.device
    dtype = b.dtype
    ones = torch.ones(3, device=device, dtype=dtype)

    candidates: list[tuple[float, torch.Tensor, dict[str, float]]] = []

    # If s == 0, the objective is linear over the simplex; any minimizer is attained at a vertex.
    if float(torch.abs(s).item()) <= eps:
        best_i = int(torch.argmin(b).item())
        p = torch.zeros(3, device=device, dtype=dtype)
        p[best_i] = 1.0
        dbg = {
            "case_id": float(best_i),
            "phi": float(b[best_i].item()),
            "p1": float(p[0].item()),
            "p2": float(p[1].item()),
            "p3": float(p[2].item()),
        }
        return p, dbg

    # Vertices
    for i in range(3):
        p = torch.zeros(3, device=device, dtype=dtype)
        p[i] = 1.0
        _raco_add_candidate_k3(candidates, p, b, H, s, case_id=float(i), simplex_tol=simplex_tol)

    # Edges: reuse the existing exact K=2 solver on each face.
    edge_specs = [
        (0, 1, 10.0),
        (0, 2, 11.0),
        (1, 2, 12.0),
    ]
    for i, j, case_id in edge_specs:
        lam, _ = _raco_solve_lambda_k2(
            b1=b[i],
            b2=b[j],
            H11=H[i, i],
            H12=H[i, j],
            H22=H[j, j],
            s=s,
            eps=eps,
        )
        p = torch.zeros(3, device=device, dtype=dtype)
        p[i] = lam
        p[j] = 1.0 - lam
        _raco_add_candidate_k3(candidates, p, b, H, s, case_id=case_id, simplex_tol=simplex_tol)

    # Interior positive-norm candidates (general pinv/KKT form).
    H_pinv = torch.linalg.pinv(H, hermitian=True)
    A = torch.dot(ones, H_pinv @ ones)
    B = torch.dot(ones, H_pinv @ b)
    C = torch.dot(b, H_pinv @ b)
    for nu in _raco_real_roots_quadratic(A, -2.0 * B, C - (s * s), eps=eps):
        rhs = (nu * ones) - b
        if float(torch.linalg.vector_norm(rhs).item()) <= eps:
            continue

        consistency_resid = rhs - (H @ (H_pinv @ rhs))
        if float(torch.linalg.vector_norm(consistency_resid).item()) > consistency_tol * max(
            1.0, float(torch.linalg.vector_norm(rhs).item())
        ):
            continue

        M = torch.zeros((4, 4), device=device, dtype=dtype)
        M[:3, :3] = H
        M[:3, 3] = -rhs
        M[3, :3] = ones
        y = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)

        sol = torch.linalg.pinv(M) @ y
        resid = (M @ sol) - y
        if float(torch.linalg.vector_norm(resid).item()) > consistency_tol:
            continue

        p = sol[:3]
        alpha = sol[3]
        if float(alpha.item()) <= pos_tol or torch.any(p <= pos_tol):
            continue
        quad = torch.dot(p, H @ p)
        if float(quad.item()) <= eps:
            continue

        phi = _raco_eval_phi_k3(p, b, H, s)
        if abs(float(phi.item()) - float(nu)) > 1e-5 * max(1.0, abs(float(phi.item())), abs(float(nu))):
            continue
        _raco_add_candidate_k3(candidates, p, b, H, s, case_id=20.0, simplex_tol=simplex_tol)

    # Interior zero-norm candidate (non-differentiable case).
    p_zero = _raco_zero_norm_interior_candidate_k3(H, pos_tol=pos_tol)
    if p_zero is not None:
        _raco_add_candidate_k3(candidates, p_zero, b, H, s, case_id=21.0, simplex_tol=simplex_tol)

    if not candidates:
        # Numerical fallback: choose the best vertex.
        best_i = int(torch.argmin(b).item())
        p = torch.zeros(3, device=device, dtype=dtype)
        p[best_i] = 1.0
        dbg = {
            "case_id": float(best_i),
            "phi": float(_raco_eval_phi_k3(p, b, H, s).item()),
            "p1": float(p[0].item()),
            "p2": float(p[1].item()),
            "p3": float(p[2].item()),
        }
        return p, dbg

    best_phi, best_p, best_dbg = min(candidates, key=lambda item: item[0])
    best_dbg = dict(best_dbg)
    best_dbg["phi"] = float(best_phi)
    return best_p, best_dbg


def shift_tokens_right(input_ids: torch.Tensor, decoder_start_token_id: int) -> torch.Tensor:
    """Shift input ids one token to the right, and pad with pad_token_id"""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    return shifted_input_ids


@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    """
    Data collator used for preference data. Inputs are dynamically padded to the maximum length of a batch if they are
    not all of the same length.

    Args:
        pad_token_id (`int`):
            Token ID to use for padding.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            Type of Tensor to return. Only `"pt"` is currently supported.

    Examples:
    ```python
    >>> from trl import DataCollatorForPreference

    >>> collator = DataCollatorForPreference(pad_token_id=0)
    >>> examples = [
    ...     {"prompt_input_ids": [1, 2, 3], "chosen_input_ids": [4, 5], "rejected_input_ids": [6]},
    ...     {"prompt_input_ids": [7, 8], "chosen_input_ids": [9, 10], "rejected_input_ids": [11, 12, 13]},
    ... ]
    >>> collator(examples)
    {'prompt_input_ids': tensor([[1, 2, 3],
                                 [0, 7, 8]]),
     'prompt_attention_mask': tensor([[1, 1, 1],
                                      [0, 1, 1]]),
     'chosen_input_ids': tensor([[ 4,  5],
                                 [ 9, 10]]),
     'chosen_attention_mask': tensor([[1, 1],
                                      [1, 1]]),
     'rejected_input_ids': tensor([[ 6,  0,  0],
                                   [11, 12, 13]]),
     'rejected_attention_mask': tensor([[1, 0, 0],
                                        [1, 1, 1]])
    }
    ```
    """

    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
            ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])
        # Optional RACO objective signs (float tensors of shape [B], values in {-1,+1})
        if "raco_s_quality" in examples[0]:
            raco_s_quality = torch.tensor([example["raco_s_quality"] for example in examples], dtype=torch.float32)
        if "raco_s_verbosity" in examples[0]:
            raco_s_verbosity = torch.tensor([example["raco_s_verbosity"] for example in examples], dtype=torch.float32)
        if "raco_s_faithfulness" in examples[0]:
            raco_s_faithfulness = torch.tensor(
                [example["raco_s_faithfulness"] for example in examples], dtype=torch.float32
            )

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps
        if "raco_s_quality" in examples[0]:
            output["raco_s_quality"] = raco_s_quality
        if "raco_s_verbosity" in examples[0]:
            output["raco_s_verbosity"] = raco_s_verbosity
        if "raco_s_faithfulness" in examples[0]:
            output["raco_s_faithfulness"] = raco_s_faithfulness
        if "token_type_ids" in examples[0]:
            token_type_ids = [torch.tensor(example["token_type_ids"]) for example in examples]
            output["token_type_ids"] = pad(token_type_ids, padding_value=0, padding_side="left")

        return output


class DPOTrainer(BaseTrainer):
    """
    Trainer for Direct Preference Optimization (DPO) method.

    This class is a wrapper around the [`transformers.Trainer`] class and inherits all of its attributes and methods.

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        ref_model ([`~transformers.PreTrainedModel`])
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation
            and loss. If no reference model is provided, the trainer will create a reference model with the same
            architecture as the model to be optimized.
        args ([`DPOConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        data_collator ([`~transformers.DataCollator`], *optional*):
            Function to use to form a batch from a list of elements of the processed `train_dataset` or `eval_dataset`.
            Will default to [`DataCollatorForPreference`].
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. DPO supports [preference](#preference) type and. The format of the samples can
            be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.BaseImageProcessor`], [`~transformers.FeatureExtractionMixin`] or [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. If `None`, the processing class is loaded from the model's name
            with [`~transformers.AutoTokenizer.from_pretrained`].
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function that will be used to compute metrics at evaluation. Must take a [`EvalPrediction`] and return
            a dictionary string to metric values. *Note* When passing TrainingArgs with `batch_eval_metrics` set to
            `True`, your compute_metrics function must take a boolean `compute_result` argument. This will be triggered
            after the last eval batch to signal that the function needs to calculate and return the global summary
            statistics rather than accumulating the batch-level statistics.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        optimizer_cls_and_kwargs (`Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]`, *optional*):
            A tuple containing the optimizer class and keyword arguments to use. Overrides `optim` and `optim_args` in
            `args`. Incompatible with the `optimizers` argument.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *optional*):
            A function that preprocess the logits right before caching them at each evaluation step. Must take two
            tensors, the logits and the labels, and return the logits once processed as desired. The modifications made
            by this function will be reflected in the predictions received by `compute_metrics`.

            Note that the labels (second parameter) will be `None` if the dataset does not have them.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "dpo"]
    _name = "DPO"
    _paper = {
        "title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        "id": "2305.18290",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{rafailov2023direct,
                title        = {{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}},
                author       = {Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn},
                year         = 2023,
                booktitle    = {Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023},
                url          = {http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html},
                editor       = {Alice Oh and Tristan Naumann and Amir Globerson and Kate Saenko and Moritz Hardt and Sergey Levine},
            }"""),
    }

    def __init__(
        self,
        model: str | nn.Module | PreTrainedModel,
        ref_model: PreTrainedModel | nn.Module | str | None = None,
        args: DPOConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: "PeftConfig | None" = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = DPOConfig(f"{model_name}-DPO")

        # Model and reference model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Special case for DeepSpeed: requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type == "DEEPSPEED":
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
        model_id = get_config_model_id(model.config)
        if isinstance(ref_model, str):
            model_init_kwargs = args.ref_model_init_kwargs or {}
            # Special case for DeepSpeed: requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type == "DEEPSPEED":
                model_init_kwargs["device_map"] = None
            ref_model = create_model_from_path(ref_model, **model_init_kwargs)
        else:
            if args.ref_model_init_kwargs is not None:
                logger.warning(
                    "You passed `ref_model_init_kwargs` to the `DPOConfig`, but your model is already instantiated. "
                    "The `ref_model_init_kwargs` will be ignored."
                )
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, you can simply omit the `ref_model` argument and it will be created for you."
            )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        # Get the pad token: if not provided, use the one from the processing class or the eos token
        # if the processing class does not have a pad token.
        pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
        self.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        if self.pad_token_id is None:
            raise ValueError(
                f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                "in the vocabulary before using it as a padding token."
            )

        # PEFT configuration and model wrapping
        model = self._prepare_peft_model(model, ref_model, peft_config, args)

        if args.generate_during_eval and not (is_wandb_available() or is_comet_available() or is_mlflow_available()):
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases, MLFlow or Comet to be installed."
                " Please install `wandb`, `mlflow` or `comet-ml` to resolve."
            )

        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_vision_model = model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.keys()
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name
        self.reference_free = args.reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or args.precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger kernel
        if args.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "You set `use_liger_kernel=True` but the liger kernel is not available. "
                    "Please install liger-kernel first: `pip install liger-kernel`"
                )
            if args.loss_type not in ["sigmoid", "apo_zero", "apo_down", "sppo_hard", "nca_pair"]:
                raise ValueError(
                    "You set `use_liger_kernel=True` but the loss type is not from `[sigmoid, apo_zero, apo_down, sppo_hard, nca_pair`. "
                    "Please set `loss_type='[sigmoid | apo_zero | apo_down | sppo_hard | nca_pair]'` to use the liger kernel."
                )
            self.dpo_loss_fn = LigerFusedLinearDPOLoss(
                ignore_index=args.label_pad_token_id,
                beta=args.beta,
                use_ref_model=not args.reference_free,
                average_log_prob=False,
                loss_type=args.loss_type,
            )
        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in DPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys are "prompt_input_ids", "chosen_input_ids", and
        # "rejected_input_ids". As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Data collator
        if data_collator is None:
            data_collator = DataCollatorForPreference(pad_token_id=self.pad_token_id)

        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.max_length = args.max_length
        self.truncation_mode = args.truncation_mode
        self.precompute_ref_log_probs = args.precompute_ref_log_probs
        self.use_logits_to_keep = args.use_logits_to_keep

        if args.padding_free:
            if model.config._attn_implementation != "flash_attention_2":
                logger.warning(
                    "Padding-free training is enabled, but the attention implementation is not set to "
                    "'flash_attention_2'. Padding-free training flattens batches into a single sequence, and "
                    "'flash_attention_2' is the only known attention mechanism that reliably supports this. Using "
                    "other implementations may lead to unexpected behavior. To ensure compatibility, set "
                    "`attn_implementation='flash_attention_2'` in the model configuration, or verify that your "
                    "attention mechanism can handle flattened sequences."
                )
            if args.per_device_train_batch_size == 1:
                logger.warning(
                    "You are using a per_device_train_batch_size of 1 with padding-free training. Using a batch size "
                    "of 1 annihilate the benefits of padding-free training. Please consider increasing the batch size "
                    "to at least 2."
                )
        self.padding_free = args.padding_free

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
        self.loss_weights = args.loss_weights
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)
        self.use_weighting = args.use_weighting
        self.aux_loss_coef = getattr(model.config, "router_aux_loss_coef", 0.0)
        if self.aux_loss_enabled and self.aux_loss_coef == 0.0:
            logger.warning(
                "You set `output_router_logits` to `True` in the model config, but `router_aux_loss_coef` is set to "
                "`0.0`, meaning the auxiliary loss will not be used. Either set `router_aux_loss_coef` to a value "
                "greater than `0.0`, or set `output_router_logits` to `False` if you don't want to use the auxiliary "
                "loss.",
            )
        for loss_type in self.loss_type:
            if (
                loss_type in ["hinge", "ipo", "bco_pair", "sppo_hard", "nca_pair", "apo_zero", "apo_down"]
                and args.label_smoothing > 0
            ):
                logger.warning(
                    f"You are using the {loss_type} loss type that does not support label smoothing. The "
                    "`label_smoothing` parameter will be ignored. Set `label_smoothing` to `0.0` to remove this "
                    "warning.",
                )
            if loss_type == "kto_pair":
                raise ValueError("Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.")

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}
        self.dataset_num_proc = args.dataset_num_proc

        # Dataset preparation
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    key: self._prepare_dataset(dataset, processing_class, args, key)
                    for key, dataset in eval_dataset.items()
                }
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        if "bco_pair" in self.loss_type:
            self.running = RunningMoments(self.accelerator)

    def _prepare_peft_model(
        self, model: PreTrainedModel, ref_model: PreTrainedModel, peft_config: Any, args: DPOConfig
    ) -> PreTrainedModel:
        """Prepares a model for PEFT training."""
        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                raise ValueError(
                    "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first "
                    "merge and unload the existing adapter, save the resulting base model, and then pass that base "
                    "model along with the new `peft_config` to the trainer."
                )

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            else:
                model = self._prepare_gradient_checkpointing(model, args)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        else:
            model = self._prepare_gradient_checkpointing(model, args)

        return model

    def _prepare_gradient_checkpointing(self, model: PreTrainedModel, args: DPOConfig):
        """Prepare the gradienting checkpointing for the model."""
        # For models that use gradient_checkpointing, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        if args.gradient_checkpointing:
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        return model

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin,
        args: DPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc nor writer_batch_size
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["writer_batch_size"] = 10

        with PartialState().main_process_first():
            # Extract prompt if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply the chat template if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            dataset = dataset.map(
                maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
            )

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                # IMPORTANT:
                # Use the *processing_class type* to choose the tokenization path.
                # Some text-only models may have a `model_type` that appears in image-text mappings,
                # which would incorrectly route us to `process_row` (expects a ProcessorMixin with `.tokenizer`).
                self.process_row if isinstance(processing_class, ProcessorMixin) else self.tokenize_row,
                remove_columns=["chosen", "rejected"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def tokenize_row(
        features: dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        """
        Tokenize a row of the dataset.

        Args:
            features (`dict[str, str]`):
                Row of the dataset, should contain the keys `"prompt"`, `"chosen"`, and `"rejected"`.
            processing_class ([`~transformers.PreTrainedTokenizerBase`]):
                Processing class used to process the data.
            max_prompt_length (`int` or `None`):
                Maximum length of the prompt sequence. If `None`, the prompt sequence is not truncated.
            max_completion_length (`int` or `None`):
                Maximum length of the completion sequences. If `None`, the completion sequences are not truncated.
            add_special_tokens (`bool`):
                Whether to add special tokens to the sequences. Typically used for encoder-decoder models. If `True`,
                the prompt sequence will have a bos token prepended and an eos token appended. In any case, the
                completion sequences will have an eos token appended.

        Returns:
            `dict[str, list[int]]`:
                Tokenized sequences with the keys `"prompt_input_ids"`, `"chosen_input_ids"`, and
                `"rejected_input_ids".

        Example:
        ```python
        >>> from transformers import GPT2Tokenizer

        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> features = {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}
        >>> DPOTrainer.tokenize_row(
        ...     features, tokenizer, max_prompt_length=3, max_completion_length=3, add_special_tokens=False
        ... )
        {'prompt_input_ids': [464, 6766, 318], 'chosen_input_ids': [4171, 50256], 'rejected_input_ids': [4077, 50256]}
        ```
        """
        tokenizer = processing_class  # the processing class is a tokenizer
        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        
        # Strip EOS token from chosen/rejected if already present (chat templates often include it)
        # This prevents double-EOS which breaks stop behavior
        chosen_text = features["chosen"]
        rejected_text = features["rejected"]
        eos_token = tokenizer.eos_token
        if eos_token:
            # Remove trailing whitespace first, then strip EOS if present
            chosen_text = chosen_text.rstrip()
            if chosen_text.endswith(eos_token):
                chosen_text = chosen_text[:-len(eos_token)]
            rejected_text = rejected_text.rstrip()
            if rejected_text.endswith(eos_token):
                rejected_text = rejected_text[:-len(eos_token)]
        
        chosen_input_ids = tokenizer(chosen_text, add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(rejected_text, add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        # Always append exactly one EOS to completions
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

    @staticmethod
    def process_row(
        features: dict[str, str],
        processing_class: PreTrainedTokenizerBase,
        max_prompt_length: int | None = None,
        max_completion_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[int]]:
        """
        Same as `tokenize_row` but for vision models. Please refer to `tokenize_row` for more information.
        """
        processor, tokenizer = processing_class, processing_class.tokenizer  # the processing class is a processor
        processed_features = processor(images=features["images"], text=features["prompt"], add_special_tokens=False)

        prompt_input_ids = processed_features["input_ids"][0]
        pixel_values = processed_features["pixel_values"][0]
        
        # Strip EOS token from chosen/rejected if already present (chat templates often include it)
        # This prevents double-EOS which breaks stop behavior
        chosen_text = features["chosen"]
        rejected_text = features["rejected"]
        eos_token = tokenizer.eos_token
        if eos_token:
            # Remove trailing whitespace first, then strip EOS if present
            chosen_text = chosen_text.rstrip()
            if chosen_text.endswith(eos_token):
                chosen_text = chosen_text[:-len(eos_token)]
            rejected_text = rejected_text.rstrip()
            if rejected_text.endswith(eos_token):
                rejected_text = rejected_text[:-len(eos_token)]
        
        chosen_input_ids = tokenizer(chosen_text, add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(rejected_text, add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
        # Always append exactly one EOS to completions
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]

        output = {
            "prompt_input_ids": prompt_input_ids,
            "pixel_values": pixel_values,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
        }

        if "pixel_attention_mask" in processed_features:
            output["pixel_attention_mask"] = processed_features["pixel_attention_mask"][0]
        if "image_sizes" in processed_features:
            output["image_sizes"] = processed_features["image_sizes"][0]
        if "token_type_ids" in processed_features:
            output["token_type_ids"] = processed_features["token_type_ids"][0]

        return output

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `DataCollatorForPreference`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
                "image_sizes",
                "token_type_ids",
                "ref_chosen_logps",
                "ref_rejected_logps",
            ]

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            batch_size = self.args.precompute_ref_batch_size or self.args.per_device_train_batch_size
            dataloader_params = {
                "batch_size": batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

                # Unnecessary cache clearing to avoid OOM
                empty_cache()
                self.accelerator.free_memory()

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(name="ref_chosen_logps", column=all_ref_chosen_logps)
            self.train_dataset = self.train_dataset.add_column(
                name="ref_rejected_logps", column=all_ref_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            batch_size = self.args.precompute_ref_batch_size or self.args.per_device_eval_batch_size
            dataloader_params = {
                "batch_size": batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            ref_chosen_logps = []
            ref_rejected_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                    (ref_chosen_logp, ref_rejected_logp)
                )
                ref_chosen_logps.append(ref_chosen_logp.cpu())
                ref_rejected_logps.append(ref_rejected_logp.cpu())

            all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
            all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="ref_chosen_logps", column=all_ref_chosen_logps)
            eval_dataset = eval_dataset.add_column(name="ref_rejected_logps", column=all_ref_rejected_logps)

            # Save calculated ref_chosen_logps and ref_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch, is_ref_model=True)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch, is_ref_model=True)
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, list | torch.LongTensor], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        """
        Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt and
        completion sequences.

        Args:
            batch (`dict[str, list | torch.LongTensor]`):
                A batch of input data. The batch must contain the following keys:

                - `"prompt_input_ids"`: Tensor of shape `(batch_size, prompt_length)` representing the prompt input
                  IDs.
                - `"chosen_input_ids"`: Tensor of shape `(batch_size, chosen_length)` representing the chosen
                  completion input IDs.
                - `"rejected_input_ids"`: Tensor of shape `(batch_size, rejected_length)` representing the rejected
                  completion input IDs.
                - `"prompt_pixel_values"` (optional): Tensor for pixel values, if available.
                - `"prompt_pixel_attention_mask"` (optional): Tensor for pixel attention masks, if available.

            padding_value (`int`):
                The padding value to use for the concatenated completion sequences (`chosen_input_ids` and
                `rejected_input_ids`).

        Returns:
            `dict[str, torch.LongTensor]`: A dictionary containing:

                - `"prompt_input_ids"`: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)`.
                - `"completion_input_ids"`: Concatenated chosen and rejected completion input IDs of shape `(2 *
                  batch_size, max_completion_length)`.
                - `"prompt_attention_mask"`: Concatenated prompt attention masks of shape `(2 * batch_size,
                  prompt_length)`.
                - `"completion_attention_mask"`: Concatenated chosen and rejected attention masks of shape `(2 *
                  batch_size, max_completion_length)`.
                - `"pixel_values"` (optional): Concatenated pixel values if `"prompt_pixel_values"` are present.
                - `"pixel_attention_mask"` (optional): Concatenated pixel attention masks if
                  `"prompt_pixel_attention_mask"` are present.

        Notes:
            The completion input IDs and attention masks are padded to the maximum completion length of the chosen or
            rejected sequences.
        """
        output = {}

        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )
        if "pixel_values" in batch:
            output["pixel_values"] = torch.cat([batch["pixel_values"], batch["pixel_values"]], dim=0)

        if "pixel_attention_mask" in batch:
            output["pixel_attention_mask"] = torch.cat(
                [batch["pixel_attention_mask"], batch["pixel_attention_mask"]], dim=0
            )
        if "image_sizes" in batch:
            output["image_sizes"] = torch.cat([batch["image_sizes"], batch["image_sizes"]], dim=0)
        if "token_type_ids" in batch:
            output["token_type_ids"] = torch.cat((batch["token_type_ids"], batch["token_type_ids"]))

        # Concatenate the chosen and rejected completions
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        return output

    def dpo_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        loss_type: str = "sigmoid",
        model_output: dict[str, torch.FloatTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.
            loss_type (`str`, defaults to `"sigmoid"`):
                The type of loss to compute. One of:
                - `"sigmoid"`: Sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
                - `"hinge"`: Hinge loss on the normalized likelihood from the
                  [SLiC](https://huggingface.co/papers/2305.10425) paper.
                - `"ipo"`: IPO loss from the [IPO](https://huggingface.co/papers/2310.12036) paper.
                - `"exo_pair"`: Pairwise EXO loss from the [EXO](https://huggingface.co/papers/2402.00856) paper.
                - `"nca_pair"`: Pairwise NCA loss from the [NCA](https://huggingface.co/papers/2402.05369) paper.
                - `"robust"`: Unbiased estimate of the DPO loss that is robust to preference noise from the [Robust
                  DPO](https://huggingface.co/papers/2403.00409) paper.
                - `"bco_pair"`: Pairwise BCO loss from the [BCO](https://huggingface.co/papers/2404.04656) paper.
                - `"sppo_hard"`: SPPO loss with hard label from the [SPPO](https://huggingface.co/papers/2405.00675)
                  paper.
                - `"aot"`: AOT loss for paired datasets from the [AOT](https://huggingface.co/papers/2406.05882) paper.
                - `"aot_pair"`: AOT loss for unpaired datasets from the [AOT](https://huggingface.co/papers/2406.05882)
                  paper.
                - `"discopop"`: DiscoPOP (a.k.a Log-Ratio Modulated Loss, LRML) loss from the
                  [DiscoPOP](https://huggingface.co/papers/2406.08414) paper.
                - `"apo_zero"`: APO-zero loss from the [APO](https://huggingface.co/papers/2408.06266) paper.
                - `"apo_down"`: APO-down loss from the [APO](https://huggingface.co/papers/2408.06266) paper.
                - `"sft"`: Negative log-likelihood loss (standard supervised fine-tuning loss).
            model_output (`dict[str, torch.FloatTensor]`, *optional*):
                The output of the model's forward pass. This is used to compute auxiliary losses if enabled.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`. The losses tensor contains the DPO
            loss for each example in the batch. The `chosen_rewards` and `rejected_rewards` tensors contain the rewards
            for the chosen and rejected responses, respectively.
        """
        device = self.accelerator.device

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps

            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the
        # labels and calculates a conservative DPO loss.
        if loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        elif loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)

        elif loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))

        elif loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)

        elif loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2

        elif loss_type == "bco_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )

        elif loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of the trainer class.
            # The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is
            # set to 1 for the winner and 0 for the loser.
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2

        elif loss_type == "nca_pair":
            chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
            rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
            losses = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )

        elif loss_type == "aot_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif loss_type == "aot":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
            delta = logratios_sorted - ref_logratios_sorted
            losses = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output
            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood
            losses = losses_chosen + losses_rejected

        elif loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output.
            # Decrease chosen likelihood and decrease rejected likelihood more
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected

        elif loss_type == "discopop":
            # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
            # This loss was discovered with LLM discovery
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            logits = logits * self.beta
            # Modulate the mixing coefficient based on the log ratio magnitudes
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            # Blend between logistic and exponential component based on log ratio modulation
            losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation

        elif loss_type == "sft":
            # SFT loss is the negative log likelihood loss on chosen responses
            # This acts as the generation loss component in MPO
            sft_loss = model_output["nll_loss"]
            # Create losses tensor with same shape as other losses (per-sample)
            batch_size = chosen_logps.shape[0]
            losses = sft_loss.expand(batch_size)
            # For SFT, we don't have preference rewards, so use zeros
            chosen_rewards = torch.zeros_like(chosen_logps)
            rejected_rewards = torch.zeros_like(rejected_logps)

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'discopop', 'apo_zero', "
                "'apo_down', 'sft']"
            )

        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards

    def _compute_loss_liger(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor]
    ) -> dict[str, torch.Tensor]:
        unwrapped_model = self.accelerator.unwrap_model(model)
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        if self.is_encoder_decoder:
            # 1. Get encoder outputs
            encoder_outputs = unwrapped_model.get_encoder()(
                concatenated_batch["prompt_input_ids"],
                attention_mask=concatenated_batch["prompt_attention_mask"],
                return_dict=True,
            )
            # 2. Prepare decoder inputs
            decoder_input_ids = shift_tokens_right(
                concatenated_batch["completion_input_ids"],
                unwrapped_model.config.decoder_start_token_id,
            )
            # 3. Get decoder outputs
            decoder_outputs = unwrapped_model.get_decoder()(
                input_ids=decoder_input_ids,
                attention_mask=concatenated_batch["completion_attention_mask"],
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=concatenated_batch["prompt_attention_mask"],
                use_cache=False,
            )
            hidden_states = decoder_outputs.last_hidden_state

            ref_hidden_states = None
            if not self.reference_free and self.ref_model is not None:
                unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)
                ref_encoder_outputs = unwrapped_ref_model.get_encoder()(
                    concatenated_batch["prompt_input_ids"],
                    attention_mask=concatenated_batch["prompt_attention_mask"],
                    return_dict=True,
                )
                ref_decoder_outputs = unwrapped_ref_model.get_decoder()(
                    input_ids=decoder_input_ids,
                    attention_mask=concatenated_batch["completion_attention_mask"],
                    encoder_hidden_states=ref_encoder_outputs.last_hidden_state,
                    encoder_attention_mask=concatenated_batch["prompt_attention_mask"],
                    use_cache=False,
                )
                ref_hidden_states = ref_decoder_outputs.last_hidden_state
            elif not self.reference_free:
                with self.null_ref_context():
                    ref_encoder_outputs = unwrapped_model.get_encoder()(
                        concatenated_batch["prompt_input_ids"],
                        attention_mask=concatenated_batch["prompt_attention_mask"],
                        return_dict=True,
                    )
                    ref_decoder_outputs = unwrapped_model.get_decoder()(
                        input_ids=decoder_input_ids,
                        attention_mask=concatenated_batch["completion_attention_mask"],
                        encoder_hidden_states=ref_encoder_outputs.last_hidden_state,
                        encoder_attention_mask=concatenated_batch["prompt_attention_mask"],
                        use_cache=False,
                    )
                    ref_hidden_states = ref_decoder_outputs.last_hidden_state

            labels = concatenated_batch["completion_input_ids"]
            loss_mask = completion_attention_mask.bool()
        else:
            # For decoder-only models
            input_ids = torch.cat(
                (concatenated_batch["prompt_input_ids"], concatenated_batch["completion_input_ids"]), dim=1
            )
            attention_mask = torch.cat(
                (concatenated_batch["prompt_attention_mask"], concatenated_batch["completion_attention_mask"]),
                dim=1,
            )
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush and truncate
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    # Flush left to reduce the memory usage
                    # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                    #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    # Flush right before truncating left, then flush left
                    # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                    #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                    attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            # Add logits_to_keep optimization
            if self.use_logits_to_keep:
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
                model_kwargs["logits_to_keep"] = logits_to_keep

            model_kwargs["output_hidden_states"] = True

            # Add padding-free training support
            if self.padding_free:
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            # Get the base model outputs (before LM head)
            if hasattr(unwrapped_model, "get_decoder") and unwrapped_model.get_decoder() is not None:
                base_model = unwrapped_model.get_decoder()
            else:
                base_attr = getattr(unwrapped_model, "base_model_prefix", self.args.base_model_attribute_name)
                base_model = getattr(unwrapped_model, base_attr, unwrapped_model)

            outputs = base_model(
                input_ids,
                use_cache=False,
                **model_kwargs,
            )
            hidden_states = outputs.last_hidden_state[:, :-1]

            # Get reference hidden states if needed
            ref_hidden_states = None
            if not self.reference_free and self.ref_model is not None:
                unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)
                if hasattr(unwrapped_ref_model, "get_decoder") and unwrapped_ref_model.get_decoder() is not None:
                    ref_base_model = unwrapped_ref_model.get_decoder()
                else:
                    ref_attr = getattr(unwrapped_ref_model, "base_model_prefix", self.args.base_model_attribute_name)
                    ref_base_model = getattr(unwrapped_ref_model, ref_attr, unwrapped_ref_model)

                ref_outputs = ref_base_model(
                    input_ids,
                    use_cache=False,
                    **model_kwargs,
                )
                ref_hidden_states = ref_outputs.last_hidden_state[:, :-1]
            elif not self.reference_free:
                if hasattr(unwrapped_model, "get_decoder") and unwrapped_model.get_decoder() is not None:
                    ref_base_model = unwrapped_model.get_decoder()
                else:
                    ref_attr = getattr(unwrapped_model, "base_model_prefix", self.args.base_model_attribute_name)
                    ref_base_model = getattr(unwrapped_model, ref_attr, unwrapped_model)
                with self.null_ref_context():
                    ref_outputs = ref_base_model(
                        input_ids,
                        use_cache=False,
                        **model_kwargs,
                    )
                    ref_hidden_states = ref_outputs.last_hidden_state[:, :-1]

            masked_input_ids = torch.where(loss_mask != 0, input_ids, self.label_pad_token_id)
            labels = masked_input_ids[:, 1:]  # Shift right for casual LM

        # Get the LM head
        lm_head = unwrapped_model.get_output_embeddings()

        # Get reference model weights if needed
        ref_weight = None
        ref_bias = None
        if not self.reference_free:
            if self.ref_model is not None:
                unwrapped_ref_model = self.accelerator.unwrap_model(self.ref_model)
                ref_lm_head = unwrapped_ref_model.get_output_embeddings()
            else:
                with self.null_ref_context():
                    ref_lm_head = unwrapped_model.get_output_embeddings()
            ref_weight = ref_lm_head.weight
            ref_bias = ref_lm_head.bias if hasattr(ref_lm_head, "bias") else None

        # Compute loss using Liger kernel
        loss_output = self.dpo_loss_fn(
            lm_head.weight,
            hidden_states,
            labels,
            bias=lm_head.bias if hasattr(lm_head, "bias") else None,
            ref_input=ref_hidden_states if not self.reference_free else None,
            ref_weight=ref_weight if not self.reference_free else None,
            ref_bias=ref_bias if not self.reference_free else None,
        )
        (
            loss,
            (chosen_logps, rejected_logps, chosen_logits_mean, rejected_logits_mean, nll_loss, *aux_outputs),
        ) = loss_output

        output = {
            "loss": loss,
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": chosen_logits_mean,
            "mean_rejected_logits": rejected_logits_mean,
            "nll_loss": nll_loss,
            "chosen_rewards": aux_outputs[0],
            "rejected_rewards": aux_outputs[1],
        }
        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor], is_ref_model: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Runs the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.

        Args:
            model:
                Model to run the forward pass on.
            batch:
                Batch of input data.
            is_ref_model:
                Whether this method is being called for the reference model. If `True`, length desensitization is not
                applied.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.pad_token_id)

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            if "token_type_ids" in concatenated_batch:
                prompt_token_type_ids = concatenated_batch["token_type_ids"]
                token_type_ids = pad_to_length(prompt_token_type_ids, input_ids.shape[1], 0)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush and truncate
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    # Flush left to reduce the memory usage
                    # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                    #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                    else:
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    # Flush right before truncating left, then flush left
                    # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                    #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                        token_type_ids = token_type_ids[:, -self.max_length :]
                    else:
                        attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    if "token_type_ids" in concatenated_batch:
                        attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                            attention_mask, input_ids, loss_mask, token_type_ids
                        )
                    else:
                        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                if "token_type_ids" in concatenated_batch:
                    attention_mask, input_ids, loss_mask, token_type_ids = flush_left(
                        attention_mask, input_ids, loss_mask, token_type_ids
                    )
                else:
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

            if "token_type_ids" in concatenated_batch:
                model_kwargs["token_type_ids"] = token_type_ids

            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
                model_kwargs["logits_to_keep"] = logits_to_keep

            model_kwargs["output_hidden_states"] = True

            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            outputs = model(input_ids, **model_kwargs)
            logits = outputs.logits

            # Offset the logits by one to align with the labels
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]

        if logits.shape[:2] != labels.shape[:2]:
            # for LLaVA, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps[:, 1:].sum(-1)

        output = {}

        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            # Only use the chosen logits for the RPO loss or SFT loss
            chosen_logits = logits[:num_examples, :-1] if not self.is_encoder_decoder else logits[:num_examples]
            chosen_labels = labels[:num_examples, :-1] if not self.is_encoder_decoder else labels[:num_examples]

            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        if "ipo" in self.loss_type:
            all_logps = all_logps / loss_mask.sum(-1)

        if self.args.ld_alpha is not None and not is_ref_model:
            # Compute response lengths based on loss_mask
            completion_lengths = loss_mask.sum(dim=1)

            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper
            public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

            seq_len = per_token_logps.size(1)
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)

            front_mask = (ld_mask & mask).float()
            rear_mask = (~ld_mask & mask).float()
            front_logps = (per_token_logps * front_mask).sum(dim=1)
            rear_logps = (per_token_logps * rear_mask).sum(dim=1)

            all_logps = front_logps + self.args.ld_alpha * rear_logps

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output

    def get_batch_loss_metrics(
        self,
        model: PreTrainedModel | nn.Module,
        batch: dict[str, list | torch.LongTensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        mode = _get_mode(self.args)

        if getattr(self.args, "raco", False) and self.args.use_liger_kernel:
            raise ValueError("RACO currently requires `use_liger_kernel=False` (needs explicit ref log-probs).")
        if mode == "amopo" and self.args.use_liger_kernel:
            raise ValueError("AMoPO static baseline currently requires `use_liger_kernel=False` (needs chosen/rejected log-probs).")

        if self.args.use_liger_kernel:
            model_output = self._compute_loss_liger(model, batch)
            losses = model_output["loss"]
            chosen_rewards = model_output["chosen_rewards"]
            rejected_rewards = model_output["rejected_rewards"]
        else:
            model_output = self.concatenated_forward(model, batch)

            # ---------------------------------------------------------
            # AMoPO (static baseline): Eq. (9)
            #
            # L = -E [ sum_k alpha_k * log sigma( (beta/|y_w|) log pi(y_w|x) - (beta/|y_l|) log pi(y_l|x) ) ]
            #
            # Here we reuse the RACO 2-objective setup and determine (y_w, y_l) per objective via signs:
            #  - quality: s=+1 always (winner=chosen)
            #  - verbosity: s derived from labels if provided, else from completion lengths (prefer shorter by default)
            # ---------------------------------------------------------
            if mode == "amopo":
                # lengths (avoid div by 0)
                chosen_len = batch["chosen_attention_mask"].sum(dim=1).to(model_output["chosen_logps"].device)
                rejected_len = batch["rejected_attention_mask"].sum(dim=1).to(model_output["chosen_logps"].device)
                chosen_len = torch.clamp(chosen_len, min=1).to(model_output["chosen_logps"].dtype)
                rejected_len = torch.clamp(rejected_len, min=1).to(model_output["chosen_logps"].dtype)

                # per-sequence scaled logp: (beta/|y|) * log pi(y|x)
                score_chosen = (self.beta * model_output["chosen_logps"]) / chosen_len
                score_rejected = (self.beta * model_output["rejected_logps"]) / rejected_len

                # signs
                s_quality = torch.ones_like(score_chosen)
                if "raco_s_quality" in batch:
                    s_quality = batch["raco_s_quality"].to(score_chosen.device).to(score_chosen.dtype)

                if "raco_s_verbosity" in batch:
                    s_verbosity = batch["raco_s_verbosity"].to(score_chosen.device).to(score_chosen.dtype)
                else:
                    if getattr(self.args, "raco_verbosity_prefer_shorter", True):
                        verbosity_prefers_chosen = chosen_len <= rejected_len
                    else:
                        verbosity_prefers_chosen = chosen_len >= rejected_len
                    s_verbosity = torch.where(verbosity_prefers_chosen, torch.ones_like(score_chosen), -torch.ones_like(score_chosen))

                # alpha_k: reuse raco_weights; enforce sum-to-1 for AMoPO baseline
                a1, a2 = (float(self.args.raco_weights[0]), float(self.args.raco_weights[1]))
                ssum = float(a1 + a2)
                if abs(ssum - 1.0) > 1e-6:
                    raise ValueError(f"AMoPO mode requires raco_weights to sum to 1, got sum={ssum} from {self.args.raco_weights}")

                # per-objective winners/losers
                # if s_k > 0 => winner=chosen else winner=rejected (ties treated as 0 => z=0)
                def _z_from_sign(s: torch.Tensor) -> torch.Tensor:
                    win = torch.where(s > 0, score_chosen, torch.where(s < 0, score_rejected, score_chosen))
                    lose = torch.where(s > 0, score_rejected, torch.where(s < 0, score_chosen, score_chosen))
                    z = win - lose
                    # for ties (s==0), set z=0 => logσ(0)=log(0.5)
                    z = torch.where(s == 0, torch.zeros_like(z), z)
                    return z

                z_q = _z_from_sign(s_quality)
                z_v = _z_from_sign(s_verbosity)

                loss_q = -F.logsigmoid(z_q)  # per-sample
                loss_v = -F.logsigmoid(z_v)
                losses = (a1 * loss_q) + (a2 * loss_v)

                # For consistency with existing logging keys
                prefix = "eval_" if train_eval == "eval" else ""
                metrics[f"{prefix}amopo/loss_quality"] = self.accelerator.gather_for_metrics(loss_q.detach()).mean().item()
                metrics[f"{prefix}amopo/loss_verbosity"] = self.accelerator.gather_for_metrics(loss_v.detach()).mean().item()
                metrics[f"{prefix}amopo/z_quality"] = self.accelerator.gather_for_metrics(z_q.detach()).mean().item()
                metrics[f"{prefix}amopo/z_verbosity"] = self.accelerator.gather_for_metrics(z_v.detach()).mean().item()
                # Match RACO/MODPO naming so dashboards can overlay runs:
                # p_k = σ(z_k) is the probability that the objective's preferred option wins.
                p_quality = torch.sigmoid(z_q)
                p_verbosity = torch.sigmoid(z_v)
                metrics[f"{prefix}p/quality"] = self.accelerator.gather_for_metrics(p_quality.detach()).mean().item()
                metrics[f"{prefix}p/verbosity"] = self.accelerator.gather_for_metrics(p_verbosity.detach()).mean().item()

                # define rewards for downstream metrics (treat "chosen" vs "rejected" in the usual way)
                chosen_rewards = score_chosen.detach()
                rejected_rewards = score_rejected.detach()

            else:
                # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
                if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
                    ref_chosen_logps = batch["ref_chosen_logps"]
                    ref_rejected_logps = batch["ref_rejected_logps"]
                else:
                    ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

                if getattr(self.args, "raco", False):
                    # ---------------------------------------------------------
                    # RACO Step (1): compute Δ and objective-specific DPO losses
                    #
                    # Δ_b = (logπ(y1|x)-logπ(y2|x)) - (logπ_ref(y1|x)-logπ_ref(y2|x))
                    # L_i = - mean_b log σ( β * s_{i,b} * Δ_b )
                    #
                    # Here y1 = chosen, y2 = rejected.
                    # Objective i=0: quality => s=+1 always.
                    # Objective i=1: verbosity => prefer shorter completion.
                    # ---------------------------------------------------------
                    # Optional: length-normalize per-sequence log-probs (avg logp) before forming delta.
                    if getattr(self.args, "raco_length_normalized", False):
                        chosen_len = batch["chosen_attention_mask"].sum(dim=1).to(model_output["chosen_logps"].device)
                        rejected_len = batch["rejected_attention_mask"].sum(dim=1).to(model_output["chosen_logps"].device)
                        chosen_len = torch.clamp(chosen_len, min=1).to(model_output["chosen_logps"].dtype)
                        rejected_len = torch.clamp(rejected_len, min=1).to(model_output["chosen_logps"].dtype)

                        pol_c = model_output["chosen_logps"] / chosen_len
                        pol_r = model_output["rejected_logps"] / rejected_len
                        policy_logratios = pol_c - pol_r
                        if self.reference_free:
                            ref_logratios = torch.zeros_like(policy_logratios)
                        else:
                            ref_c = ref_chosen_logps / chosen_len
                            ref_r = ref_rejected_logps / rejected_len
                            ref_logratios = ref_c - ref_r
                        delta = policy_logratios - ref_logratios
                    else:
                        policy_logratios = model_output["chosen_logps"] - model_output["rejected_logps"]
                        if self.reference_free:
                            ref_logratios = torch.zeros_like(policy_logratios)
                        else:
                            ref_logratios = ref_chosen_logps - ref_rejected_logps
                        delta = policy_logratios - ref_logratios

                    num_raco_objectives = len(getattr(self.args, "raco_weights", []))
                    if num_raco_objectives == 2:
                        # s_quality: from labels if provided, else default to +1.
                        if "raco_s_quality" in batch:
                            s_quality = batch["raco_s_quality"].to(delta.device).to(delta.dtype)
                        else:
                            s_quality = torch.ones_like(delta)

                        # s_verbosity: from labels if provided; otherwise derive from completion length.
                        if "raco_s_verbosity" in batch:
                            s_verbosity = batch["raco_s_verbosity"].to(delta.device).to(delta.dtype)
                            verbosity_prefers_chosen = s_verbosity > 0
                        else:
                            chosen_len = batch["chosen_attention_mask"].sum(dim=1).to(delta.device)
                            rejected_len = batch["rejected_attention_mask"].sum(dim=1).to(delta.device)
                            if getattr(self.args, "raco_verbosity_prefer_shorter", True):
                                verbosity_prefers_chosen = chosen_len <= rejected_len
                            else:
                                verbosity_prefers_chosen = chosen_len >= rejected_len
                            s_verbosity = torch.where(
                                verbosity_prefers_chosen, torch.ones_like(delta), -torch.ones_like(delta)
                            )

                        # Per-sample losses for each objective (sigmoid DPO form)
                        loss_quality = -F.logsigmoid(self.beta * s_quality * delta)
                        loss_verbosity = -F.logsigmoid(self.beta * s_verbosity * delta)
                        # Per-objective "p" (MODPO-style): probability that the objective's preferred option wins
                        p_quality = torch.sigmoid(self.beta * s_quality * delta)
                        p_verbosity = torch.sigmoid(self.beta * s_verbosity * delta)

                        wq, wv = (float(self.args.raco_weights[0]), float(self.args.raco_weights[1]))
                        losses = wq * loss_quality + wv * loss_verbosity
                    elif num_raco_objectives == 3:
                        if "raco_s_quality" in batch:
                            s_quality = batch["raco_s_quality"].to(delta.device).to(delta.dtype)
                        else:
                            s_quality = torch.ones_like(delta)

                        if "raco_s_verbosity" in batch:
                            s_verbosity = batch["raco_s_verbosity"].to(delta.device).to(delta.dtype)
                            verbosity_prefers_chosen = s_verbosity > 0
                        else:
                            chosen_len = batch["chosen_attention_mask"].sum(dim=1).to(delta.device)
                            rejected_len = batch["rejected_attention_mask"].sum(dim=1).to(delta.device)
                            if getattr(self.args, "raco_verbosity_prefer_shorter", True):
                                verbosity_prefers_chosen = chosen_len <= rejected_len
                            else:
                                verbosity_prefers_chosen = chosen_len >= rejected_len
                            s_verbosity = torch.where(
                                verbosity_prefers_chosen, torch.ones_like(delta), -torch.ones_like(delta)
                            )

                        if "raco_s_faithfulness" not in batch:
                            raise ValueError(
                                "3-objective RACO requires `raco_s_faithfulness` in the batch."
                            )
                        s_faithfulness = batch["raco_s_faithfulness"].to(delta.device).to(delta.dtype)

                        loss_quality = -F.logsigmoid(self.beta * s_quality * delta)
                        loss_verbosity = -F.logsigmoid(self.beta * s_verbosity * delta)
                        loss_faithfulness = -F.logsigmoid(self.beta * s_faithfulness * delta)
                        p_quality = torch.sigmoid(self.beta * s_quality * delta)
                        p_verbosity = torch.sigmoid(self.beta * s_verbosity * delta)
                        p_faithfulness = torch.sigmoid(self.beta * s_faithfulness * delta)

                        wq, wv, wf = (
                            float(self.args.raco_weights[0]),
                            float(self.args.raco_weights[1]),
                            float(self.args.raco_weights[2]),
                        )
                        losses = (wq * loss_quality) + (wv * loss_verbosity) + (wf * loss_faithfulness)
                    else:
                        raise ValueError(
                            f"Unsupported number of RACO objectives: {num_raco_objectives}. Expected 2 or 3."
                        )

                    # Keep reward logging consistent with standard DPO (quality-style rewards)
                    device = self.accelerator.device
                    if getattr(self.args, "raco_length_normalized", False):
                        chosen_len_rw = batch["chosen_attention_mask"].sum(dim=1).to(device)
                        rejected_len_rw = batch["rejected_attention_mask"].sum(dim=1).to(device)
                        chosen_len_rw = torch.clamp(chosen_len_rw, min=1).to(model_output["chosen_logps"].dtype)
                        rejected_len_rw = torch.clamp(rejected_len_rw, min=1).to(model_output["chosen_logps"].dtype)
                        chosen_rewards = self.beta * (
                            (model_output["chosen_logps"].to(device) / chosen_len_rw)
                            - (ref_chosen_logps.to(device) / chosen_len_rw)
                        ).detach()
                        rejected_rewards = self.beta * (
                            (model_output["rejected_logps"].to(device) / rejected_len_rw)
                            - (ref_rejected_logps.to(device) / rejected_len_rw)
                        ).detach()
                    else:
                        chosen_rewards = self.beta * (
                            model_output["chosen_logps"].to(device) - ref_chosen_logps.to(device)
                        ).detach()
                        rejected_rewards = self.beta * (
                            model_output["rejected_logps"].to(device) - ref_rejected_logps.to(device)
                        ).detach()

                    prefix = "eval_" if train_eval == "eval" else ""
                    metrics[f"{prefix}raco/loss_quality"] = (
                        self.accelerator.gather_for_metrics(loss_quality.detach()).mean().item()
                    )
                    metrics[f"{prefix}raco/loss_verbosity"] = (
                        self.accelerator.gather_for_metrics(loss_verbosity.detach()).mean().item()
                    )
                    # Use canonical metric names (match MODPO/AMoPO) so W&B overlays runs on the same plot.
                    metrics[f"{prefix}p/quality"] = self.accelerator.gather_for_metrics(p_quality.detach()).mean().item()
                    metrics[f"{prefix}p/verbosity"] = self.accelerator.gather_for_metrics(p_verbosity.detach()).mean().item()
                    if num_raco_objectives == 3:
                        metrics[f"{prefix}raco/loss_faithfulness"] = (
                            self.accelerator.gather_for_metrics(loss_faithfulness.detach()).mean().item()
                        )
                        metrics[f"{prefix}p/faithfulness"] = (
                            self.accelerator.gather_for_metrics(p_faithfulness.detach()).mean().item()
                        )
                    metrics[f"{prefix}raco/delta"] = self.accelerator.gather_for_metrics(delta.detach()).mean().item()
                    metrics[f"{prefix}raco/verbosity_prefers_chosen_rate"] = (
                        self.accelerator.gather_for_metrics(verbosity_prefers_chosen.float().detach()).mean().item()
                    )

                    # Optional: compute gradient norms for g_quality, g_verbosity, and g0.
                    if getattr(self.args, "raco_log_grad_norms", False) and train_eval == "train":
                        params = [p for p in model.parameters() if p.requires_grad]

                        def _grad_norm(grads):
                            # grads is a list of tensors or None; return L2 norm
                            sq = 0.0
                            for g in grads:
                                if g is None:
                                    continue
                                sq = sq + (g.detach().float().pow(2).sum().item())
                            return sq**0.5

                        # scalar objective losses
                        Lq = loss_quality.mean()
                        Lv = loss_verbosity.mean()
                        # compute per-objective gradients (retain graph so Trainer can still backward on combined loss)
                        gq = torch.autograd.grad(Lq, params, retain_graph=True, allow_unused=True)
                        gv = torch.autograd.grad(Lv, params, retain_graph=True, allow_unused=True)
                        # g0 = wq*gq + wv*gv
                        g0 = []
                        for a, b in zip(gq, gv, strict=False):
                            if a is None and b is None:
                                g0.append(None)
                            elif a is None:
                                g0.append(b * wv)
                            elif b is None:
                                g0.append(a * wq)
                            else:
                                g0.append(a * wq + b * wv)

                        metrics["raco/grad_norm_quality"] = _grad_norm(gq)
                        metrics["raco/grad_norm_verbosity"] = _grad_norm(gv)
                        metrics["raco/grad_norm_g0"] = _grad_norm(g0)

                else:
                    # Initialize combined losses
                    losses = 0
                    chosen_rewards = 0
                    rejected_rewards = 0

                    # Compute losses for each loss type
                    for idx, loss_type in enumerate(self.loss_type):
                        # Compute individual loss using standard DPO loss function
                        _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                            model_output["chosen_logps"],
                            model_output["rejected_logps"],
                            ref_chosen_logps,
                            ref_rejected_logps,
                            loss_type,
                            model_output,
                        )

                        # Add weighted contributions
                        weight = self.loss_weights[idx] if self.loss_weights else 1.0
                        losses = losses + _losses * weight
                        chosen_rewards = chosen_rewards + _chosen_rewards * weight
                        rejected_rewards = rejected_rewards + _rejected_rewards * weight

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with compute_loss_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics

        return loss

    def training_step(self, model: nn.Module, inputs: dict[str, Any], num_items_in_batch: int | None = None) -> torch.Tensor:
        """
        Override training_step to support RACO's CGrad/CAGrad-style gradient modification.
        For non-RACO runs, fall back to the standard Trainer implementation.
        """
        mode = _get_mode(self.args)
        # Only RACO mode uses the custom gradient update. AMoPO uses standard Trainer backward.
        if mode != "raco" or not getattr(self.args, "raco", False):
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        if getattr(self.accelerator, "scaler", None) is not None:
            raise ValueError("RACO custom gradients currently do not support fp16 GradScaler. Use bf16/fp32.")

        if self.is_deepspeed_enabled:
            raise ValueError(
                "RACO with CAGrad is not compatible with DeepSpeed. The custom gradient handling in RACO "
                "requires torch.autograd.grad() which breaks DeepSpeed's internal hooks. "
                "Please use standard multi-GPU DDP (e.g., multi_gpu accelerate config) instead."
            )

        model.train()
        inputs = self._prepare_inputs(inputs)

        num_raco_objectives = len(getattr(self.args, "raco_weights", []))

        # Compute objective losses and grads
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with compute_loss_context_manager:
            # We recompute per-objective losses here to obtain g_quality/g_verbosity, then apply the
            # RACO+CAGrad update direction (implemented by directly writing to .grad).
            model_output = self.concatenated_forward(model, inputs)
            if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
                ref_chosen_logps = inputs["ref_chosen_logps"]
                ref_rejected_logps = inputs["ref_rejected_logps"]
            else:
                ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(inputs)

            # Optional: length-normalize per-sequence log-probs (avg logp) before forming delta.
            if getattr(self.args, "raco_length_normalized", False):
                chosen_len = inputs["chosen_attention_mask"].sum(dim=1).to(model_output["chosen_logps"].device)
                rejected_len = inputs["rejected_attention_mask"].sum(dim=1).to(model_output["chosen_logps"].device)
                chosen_len = torch.clamp(chosen_len, min=1).to(model_output["chosen_logps"].dtype)
                rejected_len = torch.clamp(rejected_len, min=1).to(model_output["chosen_logps"].dtype)

                pol_c = model_output["chosen_logps"] / chosen_len
                pol_r = model_output["rejected_logps"] / rejected_len
                policy_logratios = pol_c - pol_r
                if self.reference_free:
                    ref_logratios = torch.zeros_like(policy_logratios)
                else:
                    ref_c = ref_chosen_logps / chosen_len
                    ref_r = ref_rejected_logps / rejected_len
                    ref_logratios = ref_c - ref_r
                delta = policy_logratios - ref_logratios
            else:
                policy_logratios = model_output["chosen_logps"] - model_output["rejected_logps"]
                if self.reference_free:
                    ref_logratios = torch.zeros_like(policy_logratios)
                else:
                    ref_logratios = ref_chosen_logps - ref_rejected_logps
                delta = policy_logratios - ref_logratios

            # Signs
            if "raco_s_quality" in inputs:
                s_quality = inputs["raco_s_quality"].to(delta.device).to(delta.dtype)
            else:
                s_quality = torch.ones_like(delta)

            if "raco_s_verbosity" in inputs:
                s_verbosity = inputs["raco_s_verbosity"].to(delta.device).to(delta.dtype)
                verbosity_prefers_chosen = s_verbosity > 0
            else:
                chosen_len = inputs["chosen_attention_mask"].sum(dim=1).to(delta.device)
                rejected_len = inputs["rejected_attention_mask"].sum(dim=1).to(delta.device)
                if getattr(self.args, "raco_verbosity_prefer_shorter", True):
                    verbosity_prefers_chosen = chosen_len <= rejected_len
                else:
                    verbosity_prefers_chosen = chosen_len >= rejected_len
                s_verbosity = torch.where(verbosity_prefers_chosen, torch.ones_like(delta), -torch.ones_like(delta))

            loss_quality = -F.logsigmoid(self.beta * s_quality * delta).mean()
            loss_verbosity = -F.logsigmoid(self.beta * s_verbosity * delta).mean()

            if num_raco_objectives == 2:
                wq, wv = (float(self.args.raco_weights[0]), float(self.args.raco_weights[1]))
                loss_w = (wq * loss_quality) + (wv * loss_verbosity)
            elif num_raco_objectives == 3:
                if "raco_s_faithfulness" not in inputs:
                    raise ValueError("3-objective RACO requires `raco_s_faithfulness` in the batch.")
                s_faithfulness = inputs["raco_s_faithfulness"].to(delta.device).to(delta.dtype)
                loss_faithfulness = -F.logsigmoid(self.beta * s_faithfulness * delta).mean()
                wq, wv, wf = (
                    float(self.args.raco_weights[0]),
                    float(self.args.raco_weights[1]),
                    float(self.args.raco_weights[2]),
                )
                loss_w = (wq * loss_quality) + (wv * loss_verbosity) + (wf * loss_faithfulness)
            else:
                raise ValueError(
                    f"Unsupported number of RACO objectives: {num_raco_objectives}. Expected 2 or 3."
                )

        if num_raco_objectives == 3 and not getattr(self.args, "raco_use_cagrad", True):
            delta_mean = float(delta.detach().float().mean().item())
            delta_std = float(delta.detach().float().std().item()) if delta.numel() > 1 else 0.0
            p_quality = torch.sigmoid(self.beta * s_quality * delta)
            p_verbosity = torch.sigmoid(self.beta * s_verbosity * delta)
            p_faithfulness = torch.sigmoid(self.beta * s_faithfulness * delta)
            self.store_metrics(
                {
                    "raco/loss_quality": float(loss_quality.detach().float().item()),
                    "raco/loss_verbosity": float(loss_verbosity.detach().float().item()),
                    "raco/loss_faithfulness": float(loss_faithfulness.detach().float().item()),
                    "raco/loss_w": float(loss_w.detach().float().item()),
                    "p/quality": float(p_quality.detach().float().mean().item()),
                    "p/verbosity": float(p_verbosity.detach().float().mean().item()),
                    "p/faithfulness": float(p_faithfulness.detach().float().mean().item()),
                    "raco/w_quality": float(wq),
                    "raco/w_verbosity": float(wv),
                    "raco/w_faithfulness": float(wf),
                    "raco/delta_mean": delta_mean,
                    "raco/delta_std": delta_std,
                    "raco/verbosity_prefers_chosen_rate": float(
                        verbosity_prefers_chosen.float().detach().mean().item()
                    ),
                },
                train_eval="train",
            )

            ga = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
            loss_scaled = loss_w / float(ga)
            self.accelerator.backward(loss_scaled)
            return loss_scaled.detach()

        if num_raco_objectives == 3:
            # Exact K=3 CAGrad solver: keep the existing K=2 path untouched below.
            params = [p for p in model.parameters() if p.requires_grad]
            gq = list(torch.autograd.grad(loss_quality, params, retain_graph=True, allow_unused=True))
            gv = list(torch.autograd.grad(loss_verbosity, params, retain_graph=True, allow_unused=True))
            gf = list(torch.autograd.grad(loss_faithfulness, params, retain_graph=True, allow_unused=True))

            H11 = _grads_dot(gq, gq)
            H12 = _grads_dot(gq, gv)
            H13 = _grads_dot(gq, gf)
            H22 = _grads_dot(gv, gv)
            H23 = _grads_dot(gv, gf)
            H33 = _grads_dot(gf, gf)

            H = torch.stack(
                [
                    torch.stack([H11, H12, H13]),
                    torch.stack([H12, H22, H23]),
                    torch.stack([H13, H23, H33]),
                ]
            )

            w_vec = torch.tensor(
                [float(self.args.raco_weights[0]), float(self.args.raco_weights[1]), float(self.args.raco_weights[2])],
                device=H.device,
                dtype=H.dtype,
            )
            wq, wv, wf = float(w_vec[0].item()), float(w_vec[1].item()), float(w_vec[2].item())

            b = H @ w_vec
            g0_norm_sq = torch.dot(w_vec, b)
            g0_norm = torch.sqrt(torch.clamp(g0_norm_sq, min=0.0))

            c = float(getattr(self.args, "raco_c", 0.4))
            s = (g0_norm * c)

            p_raw = None
            p_mix = None
            gp_norm = None
            dbg: dict[str, float] = {}

            if getattr(self.args, "raco_use_cagrad", True) and float(g0_norm.item()) > 0.0 and c > 0.0:
                p_raw, dbg = _raco_solve_p_k3_exact(b=b, H=H, s=s)
                p_raw = p_raw.to(device=H.device, dtype=H.dtype)
                if getattr(self.args, "raco_clip_lambda", False):
                    p_mix = torch.minimum(p_raw, w_vec)
                else:
                    p_mix = p_raw
                gp_norm_sq = torch.dot(p_mix, H @ p_mix)
                gp_norm = torch.sqrt(torch.clamp(gp_norm_sq, min=0.0))

            if p_mix is not None and gp_norm is not None and float(gp_norm.item()) > 0.0:
                scale_p = float((s / gp_norm).detach().float().item())
                a1 = float(wq + scale_p * float(p_mix[0].item()))
                a2 = float(wv + scale_p * float(p_mix[1].item()))
                a3 = float(wf + scale_p * float(p_mix[2].item()))
            else:
                a1 = float(wq)
                a2 = float(wv)
                a3 = float(wf)

            delta_mean = float(delta.detach().float().mean().item())
            delta_std = float(delta.detach().float().std().item()) if delta.numel() > 1 else 0.0
            p_quality = torch.sigmoid(self.beta * s_quality * delta)
            p_verbosity = torch.sigmoid(self.beta * s_verbosity * delta)
            p_faithfulness = torch.sigmoid(self.beta * s_faithfulness * delta)

            if p_raw is not None:
                p_raw_quality = float(p_raw[0].detach().float().item())
                p_raw_verbosity = float(p_raw[1].detach().float().item())
                p_raw_faithfulness = float(p_raw[2].detach().float().item())
            else:
                p_raw_quality = 1.0 / 3.0
                p_raw_verbosity = 1.0 / 3.0
                p_raw_faithfulness = 1.0 / 3.0

            if p_mix is not None:
                p_mix_quality = float(p_mix[0].detach().float().item())
                p_mix_verbosity = float(p_mix[1].detach().float().item())
                p_mix_faithfulness = float(p_mix[2].detach().float().item())
            else:
                p_mix_quality = 1.0 / 3.0
                p_mix_verbosity = 1.0 / 3.0
                p_mix_faithfulness = 1.0 / 3.0

            self.store_metrics(
                {
                    "raco/cagrad_solver_case": float(dbg.get("case_id", -1.0)),
                    "raco/cagrad_phi": float(dbg.get("phi", float("nan"))),
                    "raco/loss_quality": float(loss_quality.detach().float().item()),
                    "raco/loss_verbosity": float(loss_verbosity.detach().float().item()),
                    "raco/loss_faithfulness": float(loss_faithfulness.detach().float().item()),
                    "raco/loss_w": float(loss_w.detach().float().item()),
                    "raco/p_raw_quality": p_raw_quality,
                    "raco/p_raw_verbosity": p_raw_verbosity,
                    "raco/p_raw_faithfulness": p_raw_faithfulness,
                    "raco/p_mix_quality": p_mix_quality,
                    "raco/p_mix_verbosity": p_mix_verbosity,
                    "raco/p_mix_faithfulness": p_mix_faithfulness,
                    "raco/p_raw_sum": p_raw_quality + p_raw_verbosity + p_raw_faithfulness,
                    "raco/p_mix_sum": p_mix_quality + p_mix_verbosity + p_mix_faithfulness,
                    "raco/w_quality": float(wq),
                    "raco/w_verbosity": float(wv),
                    "raco/w_faithfulness": float(wf),
                    "raco/a_quality": float(a1),
                    "raco/a_verbosity": float(a2),
                    "raco/a_faithfulness": float(a3),
                    "raco/g0_norm": float(g0_norm.detach().float().item()),
                    "raco/gp_norm": float(gp_norm.detach().float().item()) if gp_norm is not None else 0.0,
                    "raco/s": float(s.detach().float().item()),
                    "raco/delta_mean": delta_mean,
                    "raco/delta_std": delta_std,
                    "raco/verbosity_prefers_chosen_rate": float(verbosity_prefers_chosen.float().detach().mean().item()),
                    "p/quality": float(p_quality.detach().float().mean().item()),
                    "p/verbosity": float(p_verbosity.detach().float().mean().item()),
                    "p/faithfulness": float(p_faithfulness.detach().float().mean().item()),
                },
                train_eval="train",
            )

            ga = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
            scale = 1.0 / float(ga)

            for p, g1_i, g2_i, g3_i in zip(params, gq, gv, gf, strict=False):
                if g1_i is None and g2_i is None and g3_i is None:
                    continue

                g_det = None
                if g1_i is not None:
                    g_det = g1_i.detach() * a1
                if g2_i is not None:
                    term = g2_i.detach() * a2
                    g_det = term if g_det is None else (g_det + term)
                if g3_i is not None:
                    term = g3_i.detach() * a3
                    g_det = term if g_det is None else (g_det + term)
                if g_det is None:
                    continue
                if scale != 1.0:
                    g_det.mul_(scale)

                if p.grad is None:
                    p.grad = g_det
                else:
                    p.grad.add_(g_det)

            return loss_w.detach() / float(ga)

        # Gradients g1,g2 (w.r.t. parameters)
        # NOTE: RACO+CAGrad can be memory-hungry if we materialize multiple full gradient vectors on GPU
        # (gq/gv, g0, gp, gp_unit, g_update). To reduce peak memory, we:
        #  - compute only the scalar dot-products needed to solve for lambda
        #  - apply the final update direction per-parameter (streaming), without building extra grad lists
        params = [p for p in model.parameters() if p.requires_grad]
        gq = list(torch.autograd.grad(loss_quality, params, retain_graph=True, allow_unused=True))
        gv = list(torch.autograd.grad(loss_verbosity, params, retain_graph=True, allow_unused=True))

        # Scalar geometry terms (K=2)
        H11 = _grads_dot(gq, gq)
        H12 = _grads_dot(gq, gv)
        H22 = _grads_dot(gv, gv)
        gq_norm = torch.sqrt(torch.clamp(H11, min=0.0))
        gv_norm = torch.sqrt(torch.clamp(H22, min=0.0))
        cos_qv = _safe_scalar_cosine(H12, gq_norm, gv_norm)
        ratio_q_over_v = _safe_scalar_ratio(gq_norm, gv_norm)
        ratio_v_over_q = _safe_scalar_ratio(gv_norm, gq_norm)
        ratio_max_over_min = _safe_scalar_ratio(torch.maximum(gq_norm, gv_norm), torch.minimum(gq_norm, gv_norm))

        wq, wv = (float(self.args.raco_weights[0]), float(self.args.raco_weights[1]))

        # g0 = wq*g1 + wv*g2  (but don't materialize it)
        # ||g0||^2 = wq^2||g1||^2 + 2 wq wv <g1,g2> + wv^2||g2||^2
        g0_norm_sq = (wq * wq) * H11 + (2.0 * wq * wv) * H12 + (wv * wv) * H22
        g0_norm = torch.sqrt(torch.clamp(g0_norm_sq, min=0.0))

        c = float(getattr(self.args, "raco_c", 0.4))
        s = (g0_norm * c)

        lam = 0.5
        gp_norm = None
        dbg = {}

        if getattr(self.args, "raco_use_cagrad", True) and float(g0_norm.item()) > 0.0 and c > 0.0:
            # b1=<g1,g0>=wq<g1,g1>+wv<g1,g2>; b2=<g2,g0>=wq<g2,g1>+wv<g2,g2>
            b1 = (wq * H11) + (wv * H12)
            b2 = (wq * H12) + (wv * H22)
            lam_raw, dbg = _raco_solve_lambda_k2(b1=b1, b2=b2, H11=H11, H12=H12, H22=H22, s=s)

            # Optional clipping: p_i = clip(p_i, 0, w_i) for each objective INDEPENDENTLY
            # This ensures the mixing coefficient doesn't exceed the weight for each objective
            # p1 (quality) = min(lam, wq), p2 (verbosity) = min(1-lam, wv)
            # After clipping, p1 + p2 may not equal 1
            if getattr(self.args, "raco_clip_lambda", False):
                p1 = min(float(lam_raw), wq)  # p_quality clipped to w_quality
                p2 = min(1.0 - float(lam_raw), wv)  # p_verbosity clipped to w_verbosity
            else:
                p1 = float(lam_raw)
                p2 = 1.0 - float(lam_raw)

            p1_t = torch.tensor(p1, device=g0_norm.device, dtype=g0_norm.dtype)
            p2_t = torch.tensor(p2, device=g0_norm.device, dtype=g0_norm.dtype)
            # ||gp||^2 for gp = p1*g1 + p2*g2
            gp_norm_sq = (p1_t * p1_t) * H11 + (2.0 * p1_t * p2_t) * H12 + (p2_t * p2_t) * H22
            gp_norm = torch.sqrt(torch.clamp(gp_norm_sq, min=0.0))

        # Coefficients for the per-parameter update:
        # g_update = g0 + s * gp / ||gp||
        #         = (wq + (s/||gp||)*p1) * g1  +  (wv + (s/||gp||)*p2) * g2
        if gp_norm is not None and float(gp_norm.item()) > 0.0:
            scale_p = float((s / gp_norm).detach().float().item())
            a1 = float(wq + scale_p * p1)
            a2 = float(wv + scale_p * p2)
        else:
            a1 = float(wq)
            a2 = float(wv)

        # Lightweight debug metrics
        delta_mean = float(delta.detach().float().mean().item())
        delta_std = float(delta.detach().float().std().item()) if delta.numel() > 1 else 0.0
        if getattr(self.args, "raco_use_cagrad", True) and float(g0_norm.item()) > 0.0 and c > 0.0:
            metrics_dict = {
                "raco/lambda_raw": float(lam_raw),
                "raco/loss_quality": float(loss_quality.detach().float().item()),
                "raco/loss_verbosity": float(loss_verbosity.detach().float().item()),
                "raco/loss_w": float(loss_w.detach().float().item()),
                "raco/g_quality_norm": float(gq_norm.detach().float().item()),
                "raco/g_verbosity_norm": float(gv_norm.detach().float().item()),
                "raco/cos_quality_verbosity": float(cos_qv.detach().float().item()),
                "raco/norm_ratio_quality_over_verbosity": float(ratio_q_over_v.detach().float().item()),
                "raco/norm_ratio_verbosity_over_quality": float(ratio_v_over_q.detach().float().item()),
                "raco/norm_ratio_larger_over_smaller": float(ratio_max_over_min.detach().float().item()),
                "raco/p_mix_quality": float(p1),
                "raco/p_mix_verbosity": float(p2),
                "raco/w_quality": float(wq),
                "raco/w_verbosity": float(wv),
                "raco/g0_norm": float(g0_norm.detach().float().item()),
                "raco/gp_norm": float(gp_norm.detach().float().item()) if gp_norm is not None else 0.0,
                "raco/s": float(s.detach().float().item()),
                "raco/h": float(dbg.get("h", float("nan"))) if isinstance(dbg, dict) else float("nan"),
                "raco/delta_mean": delta_mean,
                "raco/delta_std": delta_std,
            }
            self.store_metrics(metrics_dict, train_eval="train")
        else:
            self.store_metrics(
                {
                    "raco/lambda": 0.5,
                    "raco/loss_quality": float(loss_quality.detach().float().item()),
                    "raco/loss_verbosity": float(loss_verbosity.detach().float().item()),
                    "raco/loss_w": float(loss_w.detach().float().item()),
                    "raco/g_quality_norm": float(gq_norm.detach().float().item()),
                    "raco/g_verbosity_norm": float(gv_norm.detach().float().item()),
                    "raco/cos_quality_verbosity": float(cos_qv.detach().float().item()),
                    "raco/norm_ratio_quality_over_verbosity": float(ratio_q_over_v.detach().float().item()),
                    "raco/norm_ratio_verbosity_over_quality": float(ratio_v_over_q.detach().float().item()),
                    "raco/norm_ratio_larger_over_smaller": float(ratio_max_over_min.detach().float().item()),
                    "raco/p_mix_quality": 0.5,
                    "raco/p_mix_verbosity": 0.5,
                    "raco/w_quality": float(wq),
                    "raco/w_verbosity": float(wv),
                    "raco/g0_norm": float(g0_norm.detach().float().item()),
                    "raco/s": float(s.detach().float().item()),
                    "raco/delta_mean": delta_mean,
                    "raco/delta_std": delta_std,
                },
                train_eval="train",
            )

        # Gradient accumulation scaling (match Trainer behavior: scale loss by 1/grad_accum)
        ga = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        scale = 1.0 / float(ga)

        # Accumulate into .grad (streaming; avoid building a full g_update list)
        for p, g1_i, g2_i in zip(params, gq, gv, strict=False):
            if g1_i is None and g2_i is None:
                continue

            # g = a1*g1 + a2*g2 (a1/a2 are Python floats; operation preserves the grad tensor dtype)
            if g1_i is None:
                g_det = g2_i.detach() * (a2 * scale)
            elif g2_i is None:
                g_det = g1_i.detach() * (a1 * scale)
            else:
                # Compute into a fresh tensor (cannot be purely in-place without overwriting g1_i/g2_i)
                g_det = (g1_i.detach() * a1) + (g2_i.detach() * a2)
                if scale != 1.0:
                    g_det.mul_(scale)

            if p.grad is None:
                p.grad = g_det
            else:
                p.grad.add_(g_det)

        return loss_w.detach() / float(ga)

    def generate_from_model_and_ref(self, model, batch: dict[str, torch.LongTensor]) -> tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explicitly call generate with
        # the torch amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        with generate_context_manager:
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.pad_token_id,
            )

            # if ref_output in batch use that otherwise use the reference model
            if "ref_output" in batch:
                ref_output = batch["ref_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        ref_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.pad_token_id,
                        )
                else:
                    ref_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.pad_token_id)
        policy_output_decoded = self.processing_class.batch_decode(policy_output, skip_special_tokens=True)

        ref_output = pad_to_length(ref_output, self.max_length, self.pad_token_id)
        ref_output_decoded = self.processing_class.batch_decode(ref_output, skip_special_tokens=True)

        return policy_output_decoded, ref_output_decoded

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        with torch.no_grad(), prediction_context_manager:
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = [v for k, v in logits_dict.items() if k not in ignore_keys]
        logits = torch.tensor(logits, device=self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch. Prediction/evaluation loop, shared by
        `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.generate_from_model_and_ref(self.model, random_batch)

            table = pd.DataFrame(
                columns=["Prompt", "Policy", "Ref Model"],
                data=[
                    [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                    for prompt, pol, ref in zip(
                        random_batch_dataset["prompt"], policy_output_decoded, ref_output_decoded, strict=True
                    )
                ],
            )
            if "wandb" in self.args.report_to and self.accelerator.is_main_process:
                wandb.log({"game_log": wandb.Table(data=table)})

            if "comet_ml" in self.args.report_to:
                log_table_to_comet_experiment(
                    name="game_log.csv",
                    table=table,
                )

            if "mlflow" in self.args.report_to and self.accelerator.is_main_process:
                mlflow.log_table(data=table, artifact_file="game_log.json")

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`float`, *optional*):
                Start time of the training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs, start_time)

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
