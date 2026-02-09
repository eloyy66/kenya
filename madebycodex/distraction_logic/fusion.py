"""Fusion multimodal de estados (vision, voz, biometria)."""


class MultimodalFusion:
    """Combina estados de multiples modalidades usando pesos y confianza."""

    def __init__(self, weights=None, min_confidence=0.3):
        if weights is None:
            weights = {"vision": 0.6, "voice": 0.2, "biometrics": 0.2}
        self.weights = weights
        self.min_confidence = min_confidence

    def fuse(self, inputs):
        """
        inputs: dict {modality: {state, confidence, scores}}
        Retorna: {state, confidence, scores}
        """
        combined_scores = {}
        total_weight = 0.0

        for modality, result in inputs.items():
            weight = float(self.weights.get(modality, 0.0))
            confidence = float(result.get("confidence", 0.0))
            scores = result.get("scores") or {result.get("state", "UNKNOWN"): 1.0}

            if confidence <= 0.0 or weight <= 0.0:
                continue

            total_weight += weight * confidence
            for state, score in scores.items():
                combined_scores[state] = combined_scores.get(state, 0.0) + (score * weight * confidence)

        if total_weight <= 0.0:
            return {
                "state": "UNKNOWN",
                "confidence": 0.0,
                "scores": {},
            }

        # Normalizar
        normalized = {state: val / total_weight for state, val in combined_scores.items()}
        best_state = max(normalized, key=normalized.get)
        best_confidence = normalized[best_state]

        # Si la confianza es muy baja, devolver UNKNOWN
        if best_confidence < self.min_confidence:
            return {
                "state": "UNKNOWN",
                "confidence": best_confidence,
                "scores": normalized,
            }

        return {
            "state": best_state,
            "confidence": best_confidence,
            "scores": normalized,
        }
