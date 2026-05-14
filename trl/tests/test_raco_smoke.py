import torch

from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig


class _TinyLM(torch.nn.Module):
    """A tiny causal LM-like module that matches DPOTrainer concatenated_forward minimal expectations."""

    def __init__(self, vocab_size: int = 32, hidden: int = 16):
        super().__init__()
        self.config = type("cfg", (), {"is_encoder_decoder": False, "model_type": "tiny"})()
        self.emb = torch.nn.Embedding(vocab_size, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab_size, bias=False)
        self.warnings_issued = {}

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, **kwargs):
        x = self.emb(input_ids)
        logits = self.lm_head(x)
        out = type("out", (), {})()
        out.logits = logits
        out.last_hidden_state = x
        return out


def test_raco_training_step_smoke_cpu():
    model = _TinyLM()
    ref_model = _TinyLM()

    args = DPOConfig(
        output_dir="/tmp/trl_raco_smoke",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_length=32,
        max_prompt_length=16,
        max_completion_length=16,
        report_to=[],
        raco=True,
        raco_weights=[0.8, 0.2],
        raco_c=0.4,
        raco_use_cagrad=True,
    )

    # Minimal preference dataset (already tokenized format expected after _prepare_dataset),
    # but for this smoke we bypass dataset mapping by directly calling training_step.
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,  # not used in this smoke path
    )

    # Build a tiny batch consistent with DataCollatorForPreference output
    prompt = torch.tensor([[1, 2], [3, 4]])
    chosen = torch.tensor([[5, 6, 7], [8, 9, 0]])
    rejected = torch.tensor([[5, 6, 0], [8, 9, 10]])
    batch = {
        "prompt_input_ids": prompt,
        "prompt_attention_mask": torch.ones_like(prompt),
        "chosen_input_ids": chosen,
        "chosen_attention_mask": (chosen != 0).long(),
        "rejected_input_ids": rejected,
        "rejected_attention_mask": (rejected != 0).long(),
        "raco_s_quality": torch.tensor([1.0, -1.0]),
        "raco_s_verbosity": torch.tensor([1.0, 1.0]),
    }

    loss = trainer.training_step(trainer.model, batch)
    assert torch.is_tensor(loss)
    # Ensure grads were written
    grads = [p.grad for p in trainer.model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_amopo_static_loss_smoke_cpu():
    model = _TinyLM()
    ref_model = _TinyLM()

    args = DPOConfig(
        output_dir="/tmp/trl_amopo_smoke",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_length=32,
        max_prompt_length=16,
        max_completion_length=16,
        report_to=[],
        mode="amopo",
        raco=False,
        # reuse the same input weights as requested (sum to 1)
        raco_weights=[0.5, 0.5],
        raco_verbosity_prefer_shorter=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
    )

    prompt = torch.tensor([[1, 2], [3, 4]])
    chosen = torch.tensor([[5, 6, 7], [8, 9, 0]])
    rejected = torch.tensor([[5, 6, 0], [8, 9, 10]])
    batch = {
        "prompt_input_ids": prompt,
        "prompt_attention_mask": torch.ones_like(prompt),
        "chosen_input_ids": chosen,
        "chosen_attention_mask": (chosen != 0).long(),
        "rejected_input_ids": rejected,
        "rejected_attention_mask": (rejected != 0).long(),
        # per-dim preference signs (same as RACO dataset)
        "raco_s_quality": torch.tensor([1.0, 1.0]),
        "raco_s_verbosity": torch.tensor([1.0, -1.0]),
    }

    loss = trainer.compute_loss(trainer.model, batch)
    assert torch.is_tensor(loss)
    loss.backward()

    grads = [p.grad for p in trainer.model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


