"""Resolve a Wipe V2 UViP run and prepare its fixed CLIP task embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adavip.real_world.uvip_data import atomic_json, dataset_sources


def generate_language(path: Path, model_path: str | None) -> None:
    """Generate the historical Wipe prompt's normalized 512D embedding."""
    import torch
    from transformers import CLIPModel, CLIPTokenizerFast

    model_id = "openai/clip-vit-base-patch32"
    revision = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    prompt = "Wipe the dish using both arms."
    options = {"local_files_only": True} if model_path else {"revision": revision}
    model = CLIPModel.from_pretrained(model_path or model_id, **options).eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(model_path or model_id, **options)
    with torch.inference_mode():
        features = model.get_text_features(**tokenizer([prompt], return_tensors="pt"))
        embedding = torch.nn.functional.normalize(features, dim=-1)[0].cpu().numpy()
    if embedding.shape != (512,) or not np.isfinite(embedding).all():
        raise ValueError("Expected a finite CLIP embedding with 512 elements")
    np.save(path, embedding.astype(np.float32), allow_pickle=False)


def main() -> None:
    """Resolve input paths and write a reusable JSON config outside the checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    images = parser.add_mutually_exclusive_group(required=True)
    images.add_argument("--gelsight", type=Path, help="Directory containing the published GelSight tar files")
    images.add_argument("--hdf5", type=Path, help="Directory containing full episode HDF5 files")
    parser.add_argument("--layout", choices=("cleaned", "raw"), help="Required with --hdf5")
    parser.add_argument("--output", required=True, type=Path)
    language = parser.add_mutually_exclusive_group()
    language.add_argument("--language-embedding", type=Path)
    language.add_argument("--clip-model", help="Local directory with the pinned CLIP snapshot")
    args = parser.parse_args()
    if (args.hdf5 is None) != (args.layout is None):
        parser.error("--hdf5 and --layout must be supplied together")
    image_root = (args.gelsight or args.hdf5).resolve()
    repo = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    config_path = output / "config.json"
    if config_path.exists():
        parser.error(f"Use the existing config or select a new output directory: {config_path}")
    config = json.loads((repo / "configs/real_world/wipe_v2/uvip.json").read_text())
    embedding = args.language_embedding.resolve() if args.language_embedding else output / "language.npy"
    config.update(dataset_root=str(args.dataset.resolve()), raw_root=str(image_root),
                  wipe_v2_hdf5=args.layout or "gelsight-tar", output_root=str(output), cache_root=str(output / "images"),
                  language_embedding=str(embedding),
                  upstream_unet=str(repo / "adavip/tabero/unet.py"))
    sources = dataset_sources(config)
    missing = [source["file"] for source in sources if not (image_root / source["file"]).is_file()]
    if missing:
        parser.error(f"Missing {len(missing)} Wipe V2 image files in {image_root}: {', '.join(missing[:5])}")
    output.mkdir(parents=True, exist_ok=True)
    if not args.language_embedding and not embedding.exists():
        generate_language(embedding, args.clip_model)
    values = np.load(embedding, allow_pickle=False)
    if values.shape != (512,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid CLIP embedding: {embedding}")
    atomic_json(config_path, config)
    print(config_path)


if __name__ == "__main__":
    main()
