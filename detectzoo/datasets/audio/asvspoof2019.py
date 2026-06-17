"""ASVspoof 2019 — logical (LA) and physical (PA) access anti-spoofing corpus.

Reference:
    Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio
    Detection", Proc. Interspeech 2019. https://arxiv.org/abs/1904.05441

Challenge / licence / download:
    https://www.asvspoof.org/index2019.html

The corpus is distributed via Edinburgh DataShare (see challenge site for the
current DOI and terms). There is no unattended public HTTP download; this
loader expects a **local** extract and reads the official CM protocol files
plus ``flac/`` utterances.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from detectzoo.core.registry import register_dataset
from detectzoo.datasets.base import BaseDataset, DatasetItem

_PARTITIONS = frozenset({"train", "dev", "eval", "all"})
_TRACKS = frozenset({"LA", "PA"})


def _coerce_track(track: str) -> str:
    t = track.strip().upper()
    if t not in _TRACKS:
        raise ValueError(f"track must be 'LA' or 'PA', got {track!r}")
    return t


def _coerce_partition(partition: str) -> str:
    p = partition.strip().lower()
    if p == "val":
        p = "dev"
    if p not in _PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(_PARTITIONS)}, got {partition!r}")
    return p


def _strip_audio_stem(utt: str) -> str:
    u = utt.strip()
    lower = u.lower()
    for ext in (".flac", ".wav"):
        if lower.endswith(ext):
            return u[: -len(ext)]
    return u


def _key_to_label(key: str) -> int:
    k = key.lower().replace("-", "")
    if k == "bonafide":
        return 0
    if k in ("spoof", "spoofed"):
        return 1
    raise ValueError(f"Unknown CM trial key {key!r} (expected bonafide or spoof)")


def _parse_protocol_line(line: str) -> tuple[str, str, str, str] | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split()
    if len(parts) < 5:
        raise ValueError(f"Expected at least 5 fields in protocol line, got {len(parts)}: {line!r}")
    speaker_id, utt_id, _mid, attack_id, key = parts[0], parts[1], parts[2], parts[3], parts[4]
    return speaker_id, utt_id, attack_id, key


def _find_track_root(user_root: Path, track: str) -> Path:
    """Resolve the directory that contains the per-partition flac trees and the
    ``ASVspoof2019_{track}_cm_protocols`` folder.

    Anchoring on the **protocols** directory (rather than on the train flac
    tree) lets users load eval-only or dev-only extracts — the train split
    is large (~7 GB compressed) and is often skipped on disk-constrained
    hosts.
    """
    u = user_root.expanduser().resolve()
    proto_leaf = f"ASVspoof2019_{track}_cm_protocols"
    train_leaf = f"ASVspoof2019_{track}_train"
    dev_leaf = f"ASVspoof2019_{track}_dev"
    eval_leaf = f"ASVspoof2019_{track}_eval"

    def _looks_like_track_root(d: Path) -> bool:
        if (d / proto_leaf).is_dir():
            return True
        # Tolerate extracts that contain only flac dirs (no protocols).
        return any((d / leaf / "flac").is_dir() for leaf in (train_leaf, dev_leaf, eval_leaf))

    # User pointed straight at one of the partition leaves.
    if u.name in {train_leaf, dev_leaf, eval_leaf}:
        return u.parent

    if _looks_like_track_root(u):
        return u

    # User pointed at the dataset root (parent of LA / PA).
    for sub in (u / track, u / track.lower()):
        if _looks_like_track_root(sub):
            return sub

    raise FileNotFoundError(
        f"ASVspoof 2019 ({track}): could not locate a track root under "
        f"{user_root!s} (expanded to {u}). Expected to find either "
        f"`{proto_leaf}/` or one of "
        f"`{train_leaf}/flac`, `{dev_leaf}/flac`, `{eval_leaf}/flac`. "
        f"Point `path` at the track directory (e.g. .../{track}) or the "
        "dataset root that contains it."
    )


def _protocol_file(track_root: Path, track: str, partition: str) -> Path:
    proto_dir = track_root / f"ASVspoof2019_{track}_cm_protocols"
    if not proto_dir.is_dir():
        raise FileNotFoundError(f"ASVspoof 2019: missing protocols directory {proto_dir}")
    if partition == "train":
        names = (f"ASVspoof2019.{track}.cm.train.trn.txt",)
    elif partition == "dev":
        names = (f"ASVspoof2019.{track}.cm.dev.trl.txt",)
    else:
        names = (
            f"ASVspoof2019.{track}.cm.eval.trl.txt",
            f"ASVspoof2019.{track}.cm.eval.trl_v1.txt",
        )
    for name in names:
        p = proto_dir / name
        if p.is_file():
            return p
    tried = ", ".join(names)
    raise FileNotFoundError(f"ASVspoof 2019: no protocol file among [{tried}] in {proto_dir}")


def _flac_dir(track_root: Path, track: str, partition: str) -> Path:
    sub = "train" if partition == "train" else "dev" if partition == "dev" else "eval"
    d = track_root / f"ASVspoof2019_{track}_{sub}" / "flac"
    if not d.is_dir():
        raise FileNotFoundError(f"ASVspoof 2019: expected FLAC directory {d}")
    return d


def _resolve_audio_path(flac_dir: Path, utt_id: str) -> Path:
    stem = _strip_audio_stem(utt_id)
    for ext in (".flac", ".FLAC", ".wav", ".WAV"):
        p = flac_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return flac_dir / f"{stem}.flac"


def _partition_is_available(track_root: Path, track: str, partition: str) -> bool:
    try:
        _protocol_file(track_root, track, partition)
        _flac_dir(track_root, track, partition)
    except FileNotFoundError:
        return False
    return True


def _load_partition(
    track_root: Path,
    track: str,
    partition: str,
    *,
    skip_missing: bool,
) -> List[DatasetItem]:
    proto = _protocol_file(track_root, track, partition)
    flac_dir = _flac_dir(track_root, track, partition)
    items: List[DatasetItem] = []
    missing: List[str] = []

    with open(proto, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parsed = _parse_protocol_line(line)
            if parsed is None:
                continue
            speaker_id, utt_id, attack_id, key = parsed
            label = _key_to_label(key)
            apath = _resolve_audio_path(flac_dir, utt_id)
            if not apath.is_file():
                missing.append(str(apath))
                if skip_missing:
                    continue
                raise FileNotFoundError(
                    f"ASVspoof 2019: audio missing for trial {utt_id!r} (expected {apath})"
                )
            items.append(
                DatasetItem(
                    data=str(apath),
                    label=label,
                    metadata={
                        "modality": "audio",
                        "track": track,
                        "partition": partition,
                        "speaker_id": speaker_id,
                        "utterance_id": _strip_audio_stem(utt_id),
                        "attack_id": attack_id,
                        "class": "bonafide" if label == 0 else "spoof",
                    },
                )
            )

    if not items:
        raise RuntimeError(f"ASVspoof 2019: no trials loaded from {proto}")
    if missing and skip_missing:
        from detectzoo.utils.logger import get_logger

        get_logger(__name__).warning(
            "ASVspoof 2019: skipped %d missing files (first: %s)",
            len(missing),
            missing[0],
        )
    return items


@register_dataset("asvspoof2019", aliases=["asvspoof_2019", "asv19"])
class ASVspoof2019Dataset(BaseDataset):
    """ASVspoof 2019 CM trials (bonafide vs spoof) with per-trial FLAC paths.

    Layout (under the track root, e.g. ``.../LA``)::

        ASVspoof2019_LA_train/flac/
        ASVspoof2019_LA_dev/flac/
        ASVspoof2019_LA_eval/flac/
        ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.*.txt

    Labels follow DetectZoo convention: ``0`` = bonafide (human), ``1`` = spoof.

    Parameters
    ----------
    path
        Directory containing the **LA** or **PA** tree (the folder that holds
        ``ASVspoof2019_{track}_train``), or the parent of a single
        ``ASVspoof2019_{track}_train`` directory.
    track
        ``LA`` (logical access) or ``PA`` (physical access).
    partition
        ``\"train\"``, ``\"dev\"`` (``\"val\"`` is accepted as an alias), ``\"eval\"``,
        or ``\"all\"`` to concatenate all three partitions.
    skip_missing
        If ``True``, trials whose audio file is absent are skipped (with a log
        warning) instead of raising.
    max_samples
        Optional cap (see :class:`~detectzoo.datasets.base.BaseDataset`).
    """

    name = "asvspoof2019"
    modality = "audio"

    def __init__(
        self,
        path: str | Path,
        *,
        track: str = "LA",
        partition: str = "train",
        skip_missing: bool = False,
        max_samples: int | None = None,
    ) -> None:
        super().__init__(max_samples=max_samples)
        # Expand `~/...` for the common case of pointing at a home-dir extract.
        self.path = Path(path).expanduser()
        self.track = _coerce_track(track)
        self.partition = _coerce_partition(partition)
        self.skip_missing = skip_missing

    def _load_all(self) -> List[DatasetItem]:
        track_root = _find_track_root(self.path, self.track)
        if self.partition == "all":
            parts: Sequence[str] = ("train", "dev", "eval")
            items: List[DatasetItem] = []
            for p in parts:
                if not _partition_is_available(track_root, self.track, p):
                    continue
                items.extend(
                    _load_partition(
                        track_root,
                        self.track,
                        p,
                        skip_missing=self.skip_missing,
                    )
                )
            if not items:
                raise FileNotFoundError(
                    f"ASVspoof 2019 ({self.track}): no partitions found under {track_root}"
                )
            return items
        return _load_partition(
            track_root,
            self.track,
            self.partition,
            skip_missing=self.skip_missing,
        )
