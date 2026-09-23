"""Provenance snapshot of a transcription run (T43, ADR-017).

The snapshot lives in ``transcriptions.metadata_json`` and is the only place a
row records *how* it was decoded. With two STT backends the snapshot has to be
provider-dependent: Whisper's keys must stay exactly what history already
carries (old rows read the same), and the GigaAM-v2 CTC+LM backend records its
own knobs. No database, model or GPU is involved — the two helpers are pure.
"""

from __future__ import annotations

from diary.core.models_config import FasterWhisperProfile, GigaamCtcLmProfile
from diary.domain.transcription import service


def _whisper_profile() -> FasterWhisperProfile:
    return FasterWhisperProfile(
        provider="faster_whisper",
        model="large-v3",
        language="ru",
        device="cuda",
        compute_type="int8_float16",
        beam_size=5,
    )


def _giga_profile() -> GigaamCtcLmProfile:
    return GigaamCtcLmProfile(
        provider="gigaam2_ctc_lm",
        model_id="waveletdeboshir/gigaam-ctc-with-lm",
        revision="07d97cf8a4f178b68679cafae0600c8487877301",
        device="cuda",
        beam_width=64,
    )


def test_whisper_snapshot_keys_are_unchanged() -> None:
    """The pre-T43 key set is a compatibility contract for old history rows."""
    metadata = service._profile_metadata(_whisper_profile())
    assert metadata == {
        "profile_ref": "transcription",
        "device": "cuda",
        "compute_type": "int8_float16",
        "beam_size": 5,
    }


def test_gigaam_snapshot_records_the_decoding_knobs_and_the_pin() -> None:
    metadata = service._profile_metadata(_giga_profile())
    assert metadata == {
        "profile_ref": "transcription",
        "device": "cuda",
        "beam_width": 64,
        # ``None`` means "the checkpoint's own alpha/beta" — recorded as such,
        # not replaced by a made-up number.
        "alpha": None,
        "beta": None,
        "revision": "07d97cf8a4f178b68679cafae0600c8487877301",
    }


def test_row_model_and_language_follow_the_backend() -> None:
    """``model_name``/``language`` come from the profile, or from the backend.

    GigaAM has no ``model`` and no ``language`` field: the repository id and the
    Russian-only constant are what the NOT NULL columns get, so a row can never
    be written with an empty or wrong label.
    """
    whisper = _whisper_profile()
    assert service._profile_language(whisper) == "ru"

    giga = _giga_profile()
    assert service._profile_language(giga) == "ru"
    assert service._GIGAAM_LANGUAGE == "ru"
