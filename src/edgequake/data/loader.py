"""Dataset loading with explicit download guards.

The CWA benchmark (Tang et al. 2024, SRL, doi:10.1785/0220230393) totals
~836 GB on Hugging Face. SeisBench's chunk parameter is per-year ("_2011" ...
"_2021", "_noise1", "_noise2") BUT each year maps to a 4-year merged tar.gz —
requesting one year still downloads the whole merge file. Never trigger that
implicitly: `preview_cwa_download()` reports the real size first, and
`load_cwa()` refuses to download without `confirm=True`.

Recommended laptop path:
    1. smoke-test the evaluation harness on Iquique (~5 GB): load("iquique")
    2. decide CWA strategy (external SSD / cloud) AFTER preview_cwa_download()
"""
from __future__ import annotations

from pathlib import Path

# chunk -> merged source file on Hugging Face (mirrors seisbench/data/cwa.py)
CWA_CHUNK2FILE = {
    "_2011": "merge2011_2014.tar.gz", "_2012": "merge2011_2014.tar.gz",
    "_2013": "merge2011_2014.tar.gz", "_2014": "merge2011_2014.tar.gz",
    "_2015": "merge2015_2018.tar.gz", "_2016": "merge2015_2018.tar.gz",
    "_2017": "merge2015_2018.tar.gz", "_2018": "merge2015_2018.tar.gz",
    "_2019": "merge2019_2021.tar.gz", "_2020": "merge2019_2021.tar.gz",
    "_2021": "merge2019_2021.tar.gz",
}
CWA_REPO = "NLPLabNTUST/Merged-CWA"
CWA_NOISE_REPO = "NLPLabNTUST/Merged-CWA-Noise"


def preview_cwa_download(chunks: list[str]) -> dict:
    """Query Hugging Face for the actual byte size each chunk selection pulls.

    Returns {"files": {filename: bytes|None}, "total_gb": float|None}.
    Requires `huggingface_hub` and network access — call this BEFORE load_cwa.
    """
    from huggingface_hub import HfApi

    files = sorted({CWA_CHUNK2FILE[c] for c in chunks if c in CWA_CHUNK2FILE})
    noise = [c for c in chunks if c.startswith("_noise")]
    api = HfApi()
    sizes: dict[str, int | None] = {}
    for repo, names in ((CWA_REPO, files),
                        (CWA_NOISE_REPO, [f"noise_chunk{c[-1]}.tar.gz" for c in noise])):
        if not names:
            continue
        infos = api.get_paths_info(repo, names, repo_type="dataset")
        found = {i.path: i.size for i in infos}
        for n in names:
            sizes[n] = found.get(n)
    known = [s for s in sizes.values() if s]
    total_gb = round(sum(known) / 1e9, 1) if known else None
    return {"files": sizes, "total_gb": total_gb}


def preview_hf_download(repo: str, files: list[str]) -> dict:
    """Generic Hugging Face size preview: {"files": {name: bytes}, "total_gb"}."""
    from huggingface_hub import HfApi

    infos = HfApi().get_paths_info(repo, files, repo_type="dataset")
    sizes = {i.path: i.size for i in infos}
    known = [s for s in sizes.values() if s]
    return {"repo": repo, "files": sizes,
            "total_gb": round(sum(known) / 1e9, 2) if known else None}


def preview_ceed_download(chunks: list[str]) -> dict:
    """CEED chunk sizes (chunk = e.g. "nc2023" -> AI4EPS/quakeflow_nc
    waveform_h5/2023.h5). CEED is per-year — much lighter than CWA merges."""
    out = {"chunks": {}, "total_gb": 0.0}
    for c in chunks:
        area, year = c[:2], c[2:]
        info = preview_hf_download(f"AI4EPS/quakeflow_{area}",
                                   [f"waveform_h5/{year}.h5"])
        gb = info["total_gb"] or 0.0
        out["chunks"][c] = f"{gb} GB"
        out["total_gb"] = round(out["total_gb"] + gb, 2)
    return out


def load_cwa(chunks: list[str], sampling_rate: float = 100.0,
             component_order: str = "ZNE", confirm: bool = False):
    """Load the CWA dataset for the given chunks, refusing implicit downloads.

    Set confirm=True only after checking preview_cwa_download(chunks).
    """
    import seisbench.data as sbd

    if not confirm:
        try:
            info = preview_cwa_download(chunks)
            size = f"~{info['total_gb']} GB" if info["total_gb"] else "UNKNOWN size"
        except Exception:
            size = "UNKNOWN size (could not query Hugging Face)"
        raise RuntimeError(
            f"Refusing to download CWA chunks {chunks} ({size}) without "
            "confirm=True. Run preview_cwa_download(chunks) first and make sure "
            "you have the disk space. Note: one year still pulls its whole "
            "4-year merge file."
        )
    _apply_windows_url_fix()
    # compile_from_source=True is REQUIRED: CWA lives on Hugging Face, not in
    # the SeisBench repository, and the flag defaults to False upstream.
    return sbd.CWA(chunks=chunks, sampling_rate=sampling_rate,
                   component_order=component_order, cache="trace",
                   compile_from_source=True)


_url_fix_applied = False


def _apply_windows_url_fix():
    """SeisBench <=0.12.3 builds dataset download URLs with os.path.join,
    which uses backslashes on Windows, so the repository answers 404
    (models use a different, correct code path — that's why weights download
    fine while datasets don't). Patch the join to forward slashes.
    Idempotent; safe on all platforms.
    """
    global _url_fix_applied
    if _url_fix_applied:
        return
    import seisbench
    import seisbench.util
    from seisbench.data.base import AbstractBenchmarkDataset

    def _download_preprocessed(self, output_files, chunk):
        self.path.mkdir(parents=True, exist_ok=True)
        remote_base = str(self._remote_path()).rstrip("/")
        for file_name, output_file in zip(self._files, output_files):
            file_name = file_name.replace("$CHUNK", chunk)
            seisbench.util.download_http(
                f"{remote_base}/{file_name}", output_file,
                desc=f"Downloading {file_name}",
            )

    AbstractBenchmarkDataset._download_preprocessed = _download_preprocessed
    _url_fix_applied = True


def _with_backup_repository(factory):
    """Run factory(); on a download failure, switch SeisBench to its backup
    repository (the documented mitigation for the primary server 404-ing or
    being firewalled) and retry once."""
    try:
        return factory()
    except (FileNotFoundError, ValueError) as e:
        import seisbench

        seisbench.logger.warning(
            f"Primary SeisBench repository failed ({e}); retrying via backup "
            "repository (slower download)."
        )
        seisbench.use_backup_repository()
        return factory()


def load(name_or_path: str, sampling_rate: float = 100.0,
         component_order: str = "ZNE", chunks: list[str] | None = None, **kwargs):
    """Load any SeisBench dataset by class name ("iquique", "ethz", "ceed",
    ...) or a local SeisBench-format directory (metadata.csv + waveforms.hdf5).
    CWA must go through load_cwa() for the download guard.

    Note on sources: iquique/ethz/stead/... live on the SeisBench repository
    (DESY + GFZ mirror); INSTANCE comes straight from INGV, LEN-DB from
    Zenodo, CEED from Hugging Face (AI4EPS/quakeflow_*) — the latter work
    even when the SeisBench repository is down.
    """
    import seisbench.data as sbd

    _apply_windows_url_fix()
    common = dict(sampling_rate=sampling_rate, component_order=component_order)
    if chunks is not None:
        common["chunks"] = chunks
    p = Path(name_or_path)
    if p.exists():
        return sbd.WaveformDataset(p, **common, **kwargs)
    lname = name_or_path.lower()
    if lname in ("cwa", "cwanoise"):
        raise ValueError("Use load_cwa() for CWA — it enforces the download guard.")
    import inspect

    # classes only — dir(sbd) also lists submodules (e.g. module `iquique`
    # shadows class `Iquique` after lowercasing)
    classes = {n.lower(): n for n in dir(sbd) if inspect.isclass(getattr(sbd, n))}
    cls = classes.get(lname)
    if cls is None:
        raise ValueError(f"Unknown dataset '{name_or_path}'")
    # enable source compilation (Hugging Face / INGV / Zenodo ...) for
    # datasets that aren't hosted in the SeisBench repository
    kwargs.setdefault("compile_from_source", True)

    def factory():
        try:
            return getattr(sbd, cls)(**common, **kwargs)
        except TypeError:
            # a few classes hardcode compile_from_source in their super() call
            kw = {k: v for k, v in kwargs.items() if k != "compile_from_source"}
            return getattr(sbd, cls)(**common, **kw)

    return _with_backup_repository(factory)


def arrival_columns(metadata) -> dict[str, str]:
    """Locate P/S arrival-sample columns robustly across datasets.

    Returns e.g. {"P": "trace_p_arrival_sample", "S": "trace_s_arrival_sample"}.
    """
    cols: dict[str, str] = {}
    for phase in ("P", "S"):
        cands = [c for c in metadata.columns
                 if c.lower().endswith("_arrival_sample")
                 and f"_{phase.lower()}_" in c.lower()]
        if cands:
            # prefer the plain "trace_p_arrival_sample" over pick-specific ones
            cands.sort(key=len)
            cols[phase] = cands[0]
    return cols
