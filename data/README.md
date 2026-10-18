# Data

ROS 2 bags (`.mcap`) recorded on the Indomitus rover while driving over different terrain. Each bag contains ZED 2i RGB-D camera streams together with the rover's IMU and odometry. These signals serve as self-supervised labels (slip and chassis vibration) for training the traversability model.

Recordings are grouped into **sessions**: one session is one recording day at one location, and it has a single camera configuration.

Dataset on Hugging Face: [erc-samples/traversability-costmap](https://huggingface.co/datasets/erc-samples/traversability-costmap)

## Download

Bags are not stored in git. Use [scripts/download_data.py](../scripts/download_data.py) to fetch them into `data/raw/`:

```bash
pip install huggingface_hub
hf auth login                      # a read token is enough

python scripts/download_data.py                                  # all train + val bags
python scripts/download_data.py --split val                      # only validation bags
python scripts/download_data.py --session 2026-10-17_ucu-cinema  # one session
python scripts/download_data.py --ride-type rockfield            # one ride type
python scripts/download_data.py --bag flat_05ms_20261017_162915  # one bag
python scripts/download_data.py --revision v0.1                  # pinned dataset version
python scripts/download_data.py --dry-run                        # list only, no download
```

Filters can be repeated and combined. Bags with `split=exclude`, such as `test` rides, are skipped unless you pass `--include-excluded` or name them with `--bag`. Bags that are already downloaded are skipped. Downloaded files are checked against the sha256 in the manifest (turn this off with `--no-verify`).

## Directory structure

```
data/
├── README.md
├── manifest.csv                  # one row per bag: session, ride type, terrain, speed, duration, size, split
└── raw/
    └── <session>/                # e.g. 2026-10-17_ucu-cinema  (<YYYY-MM-DD>_<location>)
        ├── session.yaml          # date, location, lighting, surface description, notes
        ├── zed2i_record.yaml     # ZED 2i camera config used for this session
        └── <bag>/                # one ROS 2 bag
            ├── metadata.yaml
            └── <bag>_0.mcap
```

`manifest.csv` is the single source of truth for which bags exist and which split (`train` / `val` / `exclude`) each one belongs to.

## Ride types

The ride type is the first part of the bag name.

| Ride type      | Description |
|----------------|-------------|
| `test`         | Setup check. Never used for training or validation. |
| `flat`         | Mostly flat surface with small bumps and stones. |
| `singlestones` | Flat surface. The rover deliberately drives over and next to single stones. |
| `rockfield`    | Passes through the stony area from different angles (~40% stony, ~60% flat with bumps). |
| `mixed`        | All terrain types (~30% stony, ~70% flat with bumps and single stones). |

## Bag naming

```
<ride_type>[_<index>]_<speed>_<YYYYMMDD>_<HHMMSS>
```

| Part          | Required | Meaning |
|---------------|----------|---------|
| `ride_type`   | yes      | One of the ride types above. |
| `index`       | no       | Number of a repeated recording with the same ride type and speed (`_2`, `_3`, …). |
| `speed`       | yes*     | Maximum speed in tenths of m/s, followed by `ms`: `03ms` = 0.3 m/s, `05ms` = 0.5 m/s, `08ms` = 0.8 m/s. This is the speed with the joystick fully forward. The actual speed may be lower. |
| `YYYYMMDD`    | yes      | Recording date. |
| `HHMMSS`      | yes      | Recording start time (local). |

\* `test` bags have no speed, e.g. `test_20261017_162047`.

Examples:
- `flat_05ms_20261017_162915`: flat ride at max 0.5 m/s, recorded 2026-10-17 at 16:29:15
- `rockfield_08ms_20261017_171002`: rockfield ride at max 0.8 m/s
