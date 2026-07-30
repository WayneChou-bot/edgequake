"""Phase 1 CLI: evaluate a phase picker on any SeisBench dataset.

Examples (laptop):
    # smoke test on a small real dataset (~5 GB, downloads on first use)
    python scripts/eval_picking.py --dataset iquique --limit 500

    # local SeisBench-format directory (e.g. the synthetic pipeline test set)
    python scripts/eval_picking.py --dataset data/synthetic_test --limit 100

    # CWA (Taiwan) — REFUSES to download until you preview and confirm:
    python scripts/eval_picking.py --preview-cwa _2019 _2020 _2021
    python scripts/eval_picking.py --dataset cwa --chunks _2019 --confirm-download
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build_picker(spec: str, state_dict: str | None = None,
                 labels: str | None = None):
    """spec: "seisbench:phasenet:original" | "tf:PhaseNet" (repo dir).
    state_dict: optional fine-tuned .pt checkpoint layered on top."""
    kind, *rest = spec.split(":")
    if kind == "seisbench":
        from edgequake.pickers.seisbench_picker import SeisBenchPhaseNet

        return SeisBenchPhaseNet(weights=rest[1] if len(rest) > 1 else "original",
                                 state_dict_path=state_dict,
                                 labels_override=labels)
    if kind == "tf":
        from edgequake.pickers.tf_phasenet import TFPhaseNet

        return TFPhaseNet(repo_dir=rest[0] if rest else "PhaseNet")
    raise ValueError(f"Unknown picker spec: {spec}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="iquique",
                    help="SeisBench class name, 'cwa', or local dataset dir")
    ap.add_argument("--chunks", nargs="*", default=None, help="CWA chunks, e.g. _2019")
    ap.add_argument("--confirm-download", action="store_true",
                    help="required for any CWA download")
    ap.add_argument("--preview-cwa", nargs="*", default=None, metavar="CHUNK",
                    help="print CWA download size for chunks and exit")
    ap.add_argument("--preview-ceed", nargs="*", default=None, metavar="CHUNK",
                    help="print CEED download size for chunks (e.g. nc2023) and exit")
    ap.add_argument("--picker", default="seisbench:phasenet:original")
    ap.add_argument("--state-dict", default=None,
                    help="fine-tuned .pt checkpoint to load on top of --picker")
    ap.add_argument("--labels", default=None,
                    help="override output channel interpretation, e.g. PSN")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=1000, help="max traces to evaluate")
    ap.add_argument("--tolerance", type=float, default=0.5, help="match tolerance (s)")
    ap.add_argument("--threshold", type=float, default=0.3, help="operating threshold")
    ap.add_argument("--noise-dataset", default=None,
                    help="optional noise set for false-alarm rate (name or dir)")
    ap.add_argument("--out", default="outputs", help="output directory")
    args = ap.parse_args()

    from edgequake.data import loader

    if args.preview_cwa is not None:
        info = loader.preview_cwa_download(args.preview_cwa or ["_2019"])
        print(json.dumps(info, indent=2))
        return
    if args.preview_ceed is not None:
        info = loader.preview_ceed_download(args.preview_ceed or ["nc2023"])
        print(json.dumps(info, indent=2))
        return

    if args.dataset.lower() == "cwa":
        ds = loader.load_cwa(args.chunks or ["_2019"], confirm=args.confirm_download)
        ds_name = f"cwa{''.join(args.chunks or ['_2019'])}"
    else:
        if args.dataset.lower() == "ceed" and not args.confirm_download:
            info = loader.preview_ceed_download(args.chunks or ["nc2023"])
            raise SystemExit(
                f"CEED download would fetch: {json.dumps(info)}\n"
                "Re-run with --confirm-download to proceed."
            )
        ds = loader.load(args.dataset, chunks=args.chunks)
        ds_name = Path(args.dataset).name + "".join(args.chunks or [])

    picker = build_picker(args.picker, state_dict=args.state_dict,
                          labels=args.labels)
    print(f"[eval] picker={picker.name} dataset={ds_name} split={args.split} "
          f"limit={args.limit}")

    from edgequake.eval.picking import evaluate_noise, evaluate_picker
    from edgequake.eval.plots import plot_evaluation

    result = evaluate_picker(
        picker, ds, dataset_name=ds_name,
        split=args.split if args.split != "all" else "test",
        limit=args.limit, tolerance_s=args.tolerance,
    )
    if args.noise_dataset:
        noise_ds = loader.load(args.noise_dataset)
        evaluate_noise(picker, noise_ds, limit=min(args.limit, 500), result=result)

    metrics = result.metrics_at(args.threshold)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    tag = f"{ds_name}_{picker.name}".replace(":", "-")
    (out_dir / f"eval_{tag}.json").write_text(json.dumps(
        {"picker": picker.name, "dataset": ds_name, "split": args.split,
         "n_windows": len(result.caches), "tolerance_s": args.tolerance,
         "operating": metrics}, indent=2))
    plot_evaluation(result, args.threshold, str(out_dir / f"eval_{tag}.png"))

    print(json.dumps(metrics, indent=2))
    print(f"[eval] wrote eval_{tag}.json / eval_{tag}.png in {out_dir}/")


if __name__ == "__main__":
    main()
