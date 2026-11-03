"""Component 6B: NLI scoring (§9.2) — real cross-encoder NLI model.

Uses the documented `cross-encoder/nli-deberta-v3-base` (same family, lighter
than roberta-large-mnli with identical labels: entailment / contradiction /
neutral). Implements the §23.7 G7 optimization: only top-1-vs-all + adjacent
pairs (~2N instead of N²).

FIX-038: all pairs are tokenized together and scored in ONE model forward
pass (batched) instead of the transformers pipeline's per-chunk overhead —
2–4x faster on CPU. FIX-050: model loading is guarded by a double-checked
lock so concurrent first queries cannot load the weights twice.
"""
from __future__ import annotations

import threading

from app.models.evidence import Evidence, NLIRelation
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_NLI_PAIRS = 50
NLI_TRUNCATE_CHARS = 512 * 4  # §23.7 truncation


class NLIScorer:
    """Classify support / contradict / neutral between evidence pieces."""

    _LABEL_MAP = {
        "entailment": "entailment", "contradiction": "contradiction", "neutral": "neutral",
        "label_0": "entailment", "label_1": "neutral", "label_2": "contradiction",
        "LABEL_0": "entailment", "LABEL_1": "neutral", "LABEL_2": "contradiction",
    }

    def __init__(self, model_name: str | None = None) -> None:
        from app.config import get_settings

        self.model_name = model_name or get_settings().nli_model
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._id2label: dict[int, str] = {}
        self._load_lock = threading.Lock()  # FIX-050

    def _load(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        import torch
                        from transformers import (AutoModelForSequenceClassification,
                                                  AutoTokenizer)

                        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                        self._model = AutoModelForSequenceClassification.from_pretrained(
                            self.model_name, trust_remote_code=True)
                        self._model.eval()
                        self._device = "cuda" if torch.cuda.is_available() else "cpu"
                        self._model.to(self._device)
                        if hasattr(self._model.config, "id2label") and self._model.config.id2label:
                            self._id2label = {
                                int(k): str(v).lower()
                                for k, v in self._model.config.id2label.items()}
                        elif "ettinx" in self.model_name.lower():
                            self._id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

                        # Detect generic LABEL_N and remap by model family conventions
                        if self._id2label and all(
                            str(v).lower().startswith("label_")
                            for v in self._id2label.values()
                        ):
                            if "ettinx" in self.model_name.lower():
                                self._id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
                            else:
                                self._id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
                            logger.info(f"NLI label remap applied: {self._id2label}")

                        logger.info(f"NLI model loaded successfully: {self.model_name} on {self._device}")
                    except Exception as exc:
                        logger.error(f"CRITICAL: NLI model '{self.model_name}' failed to load ({exc})")
                        self._model = False
                        raise RuntimeError(f"NLI model '{self.model_name}' failed to load: {exc}") from exc
        return self._model or None

    def score_nli(self, evidence_list: list[Evidence]) -> list[NLIRelation]:
        """§9.2.3 score_nli with §23.7 pair-limit optimization (FIX-038 batched)."""
        if len(evidence_list) < 2:
            return []
        try:
            model = self._load()
        except RuntimeError:
            return []
        if not model:
            return []

        top_k = evidence_list[:5]  # §23.7 Strategy 1
        pairs: list[tuple[Evidence, Evidence]] = []
        for i in range(1, len(top_k)):          # top-1 vs all others
            pairs.append((top_k[0], top_k[i]))
        for i in range(len(top_k) - 1):         # adjacent pairs
            pairs.append((top_k[i], top_k[i + 1]))
        pairs = pairs[:MAX_NLI_PAIRS]
        if not pairs:
            return []

        premises = [p[0].chunk.content[:NLI_TRUNCATE_CHARS] for p in pairs]
        hypotheses = [p[1].chunk.content[:NLI_TRUNCATE_CHARS] for p in pairs]

        try:
            import torch

            with torch.no_grad():
                tokens = self._tokenizer(
                    premises, hypotheses,
                    padding=True, truncation=True, max_length=512,
                    return_tensors="pt",
                ).to(self._device)
                logits = model(**tokens).logits
                probs = torch.softmax(logits, dim=-1).cpu()
                labels = probs.argmax(dim=-1).tolist()
                confidences = probs.max(dim=-1).values.tolist()
        except Exception as exc:
            logger.warning(f"NLI batch inference failed: {exc}")
            return []

        relations: list[NLIRelation] = []
        for (e1, e2), label_idx, conf in zip(pairs, labels, confidences):
            raw = self._id2label.get(label_idx, "neutral")
            label = self._LABEL_MAP.get(raw, raw)
            relations.append(NLIRelation(
                evidence_a=e1.chunk.id, evidence_b=e2.chunk.id,
                relation=label, confidence=float(conf)))
        contradiction = [r for r in relations if r.relation == "contradiction"]
        if contradiction:
            logger.warning("nli_contradictions_detected", extra={"extra_fields": {
                "count": len(contradiction)}})
        return relations
