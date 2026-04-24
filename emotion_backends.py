from __future__ import annotations

import base64
import binascii
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


DEFAULT_MICROEXP_PROJECT_ROOT = Path(r"D:\Study\srtp\表情识别")
DEFAULT_REGULAR_CHECKPOINT = Path("artifacts/checkpoints/best_model.pt")
DEFAULT_STRICT_CHECKPOINT = Path("artifacts/strict_checkpoints/best_model.pt")

LABEL_ALIASES = {
    "anger": "angry",
    "happiness": "happy",
}


def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode a base64 or data-URL image payload into a BGR image."""
    encoded = (image_data or "").strip()
    if not encoded:
        raise ValueError("Missing image payload")

    if encoded.startswith("data:image"):
        try:
            encoded = encoded.split(",", 1)[1]
        except IndexError as exc:
            raise ValueError("Invalid data URL payload") from exc

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Failed to decode image payload") from exc

    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to parse image payload")
    return image


class SessionRecognizerPool:
    """Keep recognizer state isolated per session for realtime smoothing/windowing."""

    def __init__(self, factory: Callable[[], Any], session_ttl_seconds: int = 300) -> None:
        self.factory = factory
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def process(
        self,
        session_id: str,
        frame_bgr: np.ndarray,
        configure: Callable[[Any], None] | None = None,
    ) -> dict[str, Any]:
        entry = self._get_or_create_entry(session_id)
        recognizer = entry["recognizer"]
        recognizer_lock = entry["lock"]

        with recognizer_lock:
            if configure is not None:
                configure(recognizer)
            result = recognizer.process_frame(frame_bgr)

        with self._lock:
            entry["last_seen"] = time.time()

        return result

    def _get_or_create_entry(self, session_id: str) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._cleanup_locked(now)
            entry = self._sessions.get(session_id)
            if entry is None:
                entry = {
                    "recognizer": self.factory(),
                    "lock": threading.Lock(),
                    "last_seen": now,
                }
                self._sessions[session_id] = entry
            return entry

    def _cleanup_locked(self, now: float) -> None:
        stale_ids = [
            session_id
            for session_id, entry in self._sessions.items()
            if now - float(entry["last_seen"]) > self.session_ttl_seconds
        ]
        for session_id in stale_ids:
            self._sessions.pop(session_id, None)


class MicroExpressionBackend:
    """Adapter for the external micro-expression project."""

    def __init__(
        self,
        default_emotion_scores: dict[str, float],
        project_root: str | Path | None = None,
        mode: str = "regular",
        device: str = "auto",
        session_ttl_seconds: int = 300,
        regular_checkpoint: str | Path | None = None,
        strict_checkpoint: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.default_emotion_scores = dict(default_emotion_scores)
        self.project_root = Path(project_root or DEFAULT_MICROEXP_PROJECT_ROOT).resolve()
        self.mode = (mode or "regular").strip().lower()
        self.device = device
        self.session_ttl_seconds = session_ttl_seconds
        self.regular_checkpoint = Path(regular_checkpoint or DEFAULT_REGULAR_CHECKPOINT)
        self.strict_checkpoint = Path(strict_checkpoint or DEFAULT_STRICT_CHECKPOINT)
        self.logger = logger or logging.getLogger(__name__)

        self._init_lock = threading.Lock()
        self._initialized = False
        self._available = False
        self._last_error: str | None = None
        self._pool: SessionRecognizerPool | None = None

    @property
    def available(self) -> bool:
        self._ensure_initialized()
        return self._available

    @property
    def last_error(self) -> str | None:
        self._ensure_initialized()
        return self._last_error

    def analyze_image(
        self,
        image_data: str,
        session_id: str,
        motion_threshold: float | None = None,
        confidence_threshold: float | None = None,
        analysis_stride: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if not self._available or self._pool is None:
            raise RuntimeError(self._last_error or "Micro-expression backend is unavailable")

        frame_bgr = decode_base64_image(image_data)
        configure = None
        if self.mode == "strict":
            configure = lambda recognizer: self._configure_strict_recognizer(  # noqa: E731
                recognizer=recognizer,
                motion_threshold=motion_threshold,
                confidence_threshold=confidence_threshold,
                analysis_stride=analysis_stride,
            )

        result = self._pool.process(
            session_id=session_id,
            frame_bgr=frame_bgr,
            configure=configure,
        )
        return self._normalize_result(result)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return

            try:
                if self.mode not in {"regular", "strict"}:
                    raise ValueError(f"Unsupported micro-expression mode: {self.mode}")

                if not self.project_root.exists():
                    raise FileNotFoundError(
                        f"Micro-expression project not found: {self.project_root}"
                    )

                src_root = self.project_root / "src"
                if not src_root.exists():
                    raise FileNotFoundError(
                        f"Micro-expression source directory not found: {src_root}"
                    )

                src_root_str = str(src_root)
                if src_root_str not in sys.path:
                    sys.path.insert(0, src_root_str)

                from microexp_recognition import (  # type: ignore
                    RealtimeMicroExpressionRecognizer,
                    StrictRealtimeMicroExpressionRecognizer,
                )

                checkpoint_path = self._resolve_checkpoint_path()
                recognizer_cls = (
                    RealtimeMicroExpressionRecognizer
                    if self.mode == "regular"
                    else StrictRealtimeMicroExpressionRecognizer
                )

                self._pool = SessionRecognizerPool(
                    factory=lambda: recognizer_cls(
                        checkpoint_path=checkpoint_path,
                        device=self.device,
                    ),
                    session_ttl_seconds=self.session_ttl_seconds,
                )
                self._available = True
                self.logger.info(
                    "微表情识别后端已启用: mode=%s, checkpoint=%s",
                    self.mode,
                    checkpoint_path,
                )
            except Exception as exc:
                self._available = False
                self._last_error = str(exc)
                self.logger.warning("微表情识别后端初始化失败: %s", exc)
            finally:
                self._initialized = True

    def _resolve_checkpoint_path(self) -> Path:
        relative_path = (
            self.regular_checkpoint if self.mode == "regular" else self.strict_checkpoint
        )
        checkpoint_path = relative_path
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.project_root / checkpoint_path
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    def _normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        face_detected = result.get("face_bbox") is not None
        normalized_scores = self._normalize_scores(result.get("scores") or {})

        dominant_emotion = self._normalize_label(result.get("label"))
        if not dominant_emotion and normalized_scores:
            dominant_emotion = max(normalized_scores.items(), key=lambda item: item[1])[0]

        if not face_detected:
            dominant_emotion = "neutral"
            normalized_scores = dict(self.default_emotion_scores)
        elif not normalized_scores:
            normalized_scores = dict(self.default_emotion_scores)
            dominant_emotion = dominant_emotion or "neutral"
        else:
            dominant_emotion = dominant_emotion or "neutral"

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_scores": normalized_scores,
            "face_detected": face_detected,
            "backend": "microexp",
            "backend_mode": self.mode,
            "confidence": float(result.get("confidence", 0.0)),
            "motion_score": float(result.get("motion_score", 0.0)),
            "event_active": bool(result.get("event_active", False)),
        }

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        max_value = max(float(value) for value in scores.values())
        multiplier = 100.0 if max_value <= 1.00001 else 1.0
        normalized: dict[str, float] = {}

        for label, value in scores.items():
            normalized_label = self._normalize_label(label)
            scaled_value = float(value) * multiplier
            normalized[normalized_label] = normalized.get(normalized_label, 0.0) + scaled_value

        for label in self.default_emotion_scores:
            normalized.setdefault(label, 0.0)

        total = sum(normalized.values())
        if total <= 0:
            return {}

        return {label: float(score) for label, score in normalized.items()}

    def _normalize_label(self, label: Any) -> str:
        raw_label = str(label or "").strip().lower()
        if not raw_label:
            return ""
        return LABEL_ALIASES.get(raw_label, raw_label)

    @staticmethod
    def _configure_strict_recognizer(
        recognizer: Any,
        motion_threshold: float | None,
        confidence_threshold: float | None,
        analysis_stride: int | None,
    ) -> None:
        if motion_threshold is not None:
            recognizer.config.motion_threshold = float(motion_threshold)
        if confidence_threshold is not None:
            recognizer.config.confidence_threshold = float(confidence_threshold)
        if analysis_stride is not None:
            recognizer.config.analysis_stride = max(1, int(analysis_stride))
