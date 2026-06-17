"""ImBD — Imitate Before Detect.

Reference:
    Chen et al., "Imitate Before Detect: Aligning Machine Stylistic
    Preference for Machine-Revised Text Detection", AAAI 2025.

ImBD first fine-tunes a causal LM (GPT-Neo-2.7B) with **Style
Preference Optimization (SPO)** to learn machine writing preferences,
then uses the analytic sampling discrepancy (same formula as
Fast-DetectGPT) computed by the SPO-aligned model as the detection
criterion.

For best results, load the pre-trained SPO checkpoint from
HuggingFace (``"xyzhu1225/ImBD-inference"``).  The checkpoint is a
PEFT/LoRA adapter on top of ``EleutherAI/gpt-neo-2.7B``.

Without the SPO checkpoint the detector still works (it degrades to
plain Fast-DetectGPT analytic on the base model), which is useful for
API-level testing.
"""

from __future__ import annotations

from typing import Any

import torch

from detectzoo.core.base import DetectionResult
from detectzoo.core.registry import register_detector
from detectzoo.detectors.text.base import BaseTextDetector
from detectzoo.utils.logger import get_logger

logger = get_logger(__name__)


def _sampling_discrepancy_analytic(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Analytic sampling discrepancy (white-box, single model).

    When the scoring model and reference model are the same, the
    discrepancy simplifies to:

        d = (LL(x) - E_p[LL]) / sqrt(Var_p[LL])

    where p is the model's own distribution.  This is the same formula
    used by Fast-DetectGPT's analytic variant.
    """
    if labels.ndim == logits.ndim - 1:
        labels = labels.unsqueeze(-1)

    lprobs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    log_likelihood = lprobs.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs * lprobs).sum(dim=-1)
    var_ref = (probs * lprobs.square()).sum(dim=-1) - mean_ref.square()

    ll_sum = log_likelihood.sum(dim=-1)
    mean_sum = mean_ref.sum(dim=-1)
    std_sum = var_ref.sum(dim=-1).clamp(min=1e-10).sqrt()

    discrepancy = (ll_sum - mean_sum) / std_sum
    return float(discrepancy)


@register_detector("imbd")
class ImBDDetector(BaseTextDetector):
    """ImBD detector — SPO-aligned analytic discrepancy.

    Parameters:
        model_name: HuggingFace model or local path. Default is the SPO-trained
            checkpoint from HuggingFace ``"xyzhu1225/ImBD-inference"``
            (PEFT/LoRA adapter atop GPT-Neo-2.7B).
        use_peft: Whether to load the model as a PEFT adapter via
            ``AutoPeftModelForCausalLM``.  Set ``True`` when using the
            ImBD inference checkpoint.  Default ``True``.
        threshold: Decision boundary on the discrepancy score.
        device: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "xyzhu1225/ImBD",
        use_peft: bool = True,
        threshold: float = 0.0,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, threshold=threshold, device=device, **kwargs)
        self.adapter_name = "xyzhu1225/ImBD-inference"
        self.base_model_name = "EleutherAI/gpt-neo-2.7B"
        self.use_peft = use_peft

    def _load_model(self) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading ImBD model '%s' (peft=%s) …", self.model_name, self.use_peft)

        if self.use_peft:
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name, device_map=self._device
            )

            self._model = PeftModel.from_pretrained(
                base_model,
                self.adapter_name,
            ).to(self._device)
        else:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(repo_id=self.model_name, filename="model.bin")

            config = AutoConfig.from_pretrained(self.base_model_name)
            model = AutoModelForCausalLM.from_config(config).to(torch.float16)

            state_dict = torch.load(weights_path, map_location="cpu")
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.float16)

            model.load_state_dict(state_dict, strict=False)
            model.to(self._device)

            self._model = model

        self._tokenizer = AutoTokenizer.from_pretrained(self.adapter_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model.eval()

    @torch.no_grad()
    def predict(self, input_data: Any) -> DetectionResult:
        text = self._normalise_input(input_data)

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_token_type_ids=False,
        ).to(self._device)

        labels = enc.input_ids[:, 1:]
        logits = self.model(**enc).logits[:, :-1]

        score = _sampling_discrepancy_analytic(logits, labels)

        return self._make_result(score)
