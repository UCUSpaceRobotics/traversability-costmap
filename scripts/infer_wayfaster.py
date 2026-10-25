#!/usr/bin/env python3
"""Run WayFASTER on one validated make-npz clip; no training or label generation."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'preprocessing'))
sys.path.insert(0, str(ROOT / 'wayfaster'))
from src.output import staged_recording
from src.transforms import validate_transform
from src.validation import ValidationLimits, validate_sequence

INPUT_KEYS = ('color_img', 'pcloud', 'intrinsics', 'extrinsics')


def load_inputs(path, config):
    """Validate the single-clip tensor contract before allocating GPU memory."""
    frames = int(config['TIME_LENGTH'])
    width, height = config['INPUT_SIZE']
    nx, ny, nz = [int(round((b[1]-b[0])/b[2])) for b in
                  (config['GRID_BOUNDS'][k] for k in ('xbound', 'ybound', 'zbound'))]
    shapes = dict(color_img=(1, frames, 3, height, width), pcloud=(1, frames, nz, nx, ny),
                  intrinsics=(1, frames, 3, 3), extrinsics=(1, frames, 4, 4))
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in (*INPUT_KEYS, 'timestamps_ns')}
    for key, shape in shapes.items():
        if np.iscomplexobj(arrays[key]):
            raise ValueError(f'{key}: expected real-valued inputs')
        arrays[key] = arrays[key].astype(np.float32)
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'{key}: expected finite tensor with shape {shape}, got {arrays[key].shape}')
    for key in ('color_img', 'pcloud'):
        if np.any((arrays[key] < 0) | (arrays[key] > 1)):
            raise ValueError(f'{key}: expected values in [0,1]')
    stamps = arrays['timestamps_ns']
    if stamps.shape != (frames,) or not np.issubdtype(stamps.dtype, np.integer):
        raise ValueError('timestamps_ns must contain one integer acquisition timestamp per frame')
    validate_sequence(stamps, 0, frames, frames, ValidationLimits().max_frame_gap)
    if np.any(np.abs(np.linalg.det(arrays['intrinsics'])) < 1e-8):
        raise ValueError('Singular camera intrinsics')
    for transform in arrays['extrinsics'][0]:
        validate_transform(transform)
    return arrays


def run_inference(args):
    # Reuse conversion's atomic publication: neither partial nor existing results
    # are exposed/overwritten, including when model loading or inference fails.
    with staged_recording(args.output_dir) as stage:
        import torch
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from wayfaster.models.traversability_net import TravNet

        config = yaml.safe_load(args.config.read_text())['MODEL']
        if not config['PREDICT_DEPTH']:
            raise ValueError('This NPZ workflow requires MODEL.PREDICT_DEPTH: true')
        arrays = load_inputs(args.input_npz, config)
        device_name = args.device
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device_name == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA/ROCm requested, but PyTorch cannot access a GPU; use --device cpu')
        device = torch.device(device_name)

        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state_dict'), dict):
            raise ValueError('Checkpoint must contain a tensor-only state_dict dictionary')
        weights = checkpoint['state_dict']
        if not weights or any(not isinstance(key, str) or not torch.is_tensor(value)
                              for key, value in weights.items()):
            raise ValueError('Checkpoint state_dict must contain named tensors only')
        if any(key.startswith('model.') for key in weights):
            extra = [key for key in weights if not key.startswith('model.') and key != 'gamma']
            if extra:
                raise ValueError(f'Unexpected non-model checkpoint entries: {extra}')
            weights = {key.removeprefix('model.'): value for key, value in weights.items()
                       if key.startswith('model.')}
        model = TravNet(config['GRID_BOUNDS'], config['INPUT_SIZE'],
                        downsample=config['DOWNSAMPLE'], image_dim=config['LATENT_DIM'],
                        temporal_length=config['TIME_LENGTH'], predict_depth=config['PREDICT_DEPTH'],
                        fuse_pcloud=config['FUSE_PCLOUD'])
        model.load_state_dict(weights, strict=True)
        model = model.to(device).eval()
        inputs = [torch.from_numpy(arrays[key]).to(device) for key in INPUT_KEYS]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            prediction = model(*inputs)[0]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        prediction = prediction.cpu().numpy()
        expected = (1, 2, *arrays['pcloud'].shape[-2:])
        if (prediction.shape != expected or not np.isfinite(prediction).all() or
                np.any((prediction < 0) | (prediction > 1))):
            raise ValueError(f'Invalid traversability output; expected {expected} with values in [0,1]')
        np.savez_compressed(stage / 'prediction.npz', traversability=prediction,
                            timestamps_ns=arrays['timestamps_ns'])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].imshow(arrays['color_img'][0, -1].transpose(1, 2, 0))
        axes[0].set_title('Last RGB frame'); axes[0].axis('off')
        for i, title in enumerate(('Linear mu', 'Angular nu')):
            image = axes[i+1].imshow(prediction[0, i], vmin=0, vmax=1, cmap='RdYlGn', origin='upper')
            axes[i+1].set(title=title, xlabel='Y voxel index (model order)', ylabel='X voxel index (model order)')
            fig.colorbar(image, ax=axes[i+1])
        fig.tight_layout(); fig.savefig(stage / 'prediction.png', dpi=150); plt.close(fig)
        summary = dict(input=str(args.input_npz.resolve()), checkpoint=str(args.checkpoint.resolve()),
                       config=str(args.config.resolve()), grid_bounds=config['GRID_BOUNDS'],
                       device=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                       inference_seconds=elapsed, shape=list(prediction.shape),
                       note='Inference only; no fine-tuning. Maps are in raw model index order.')
        (stage / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-npz', type=Path, required=True, help='Output from preprocess_wayfaster.py make-npz')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing results are never overwritten')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'wayfaster/pretrained/wayfaster_weights_only.pt',
                        help='Tensor-only checkpoint with state_dict; loaded with weights_only=True')
    parser.add_argument('--config', type=Path, default=ROOT / 'wayfaster/configs/temporal_model.yaml')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto', help='cuda also selects ROCm GPUs')
    args = parser.parse_args()
    try:
        summary = run_inference(args)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, pickle.UnpicklingError) as exc:
        parser.exit(2, f'[ERROR] {exc}\n')
    print(f"Saved {summary['shape']} prediction to {args.output_dir} "
          f"({summary['device']}, {summary['inference_seconds']:.3f} s)")


if __name__ == '__main__':
    main()
