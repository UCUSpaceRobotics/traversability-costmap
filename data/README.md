# Data

ROS 2 bags (`.mcap`) recorded on the Indomitus rover while driving over different terrain. Each bag contains ZED 2i RGB-D camera streams together with the rover's IMU and odometry. These signals serve as self-supervised labels (slip and chassis vibration) for training the traversability model.

Recordings are grouped into **sessions**: one session is one recording day at one location, and it has a single camera configuration.

Dataset on Hugging Face: [erc-samples/traversability-costmap](https://huggingface.co/datasets/erc-samples/traversability-costmap)


## Usage

Run all commands from the root of the repository.

### Prerequisites

Create virtual environment:
```bash
python3 -m venv .venv
```

Source it:
```bash
source .venv/bin/activate
```

Install requirements:
```bash
pip install -r requirements.txt
```

Login into Hugging Face:
```bash
hf auth login
```

Paste your token which you can generate by going to [https://huggingface.co/](https://huggingface.co/), clicking on your profile icon at the top right corner -> **Access Tokens** -> **Create new token**. For downloading dataset reading token is enough.

### Download

Bags are not stored in git. Use [scripts/download_data.py](../scripts/download_data.py) to fetch them into `data/raw/`:

```bash
python scripts/download_data.py                                  # all train + val bags
python scripts/download_data.py --split val                      # only validation bags
python scripts/download_data.py --session 2026-10-17_ucu-cinema  # one session
python scripts/download_data.py --ride-type mixed                # one ride type
python scripts/download_data.py --bag flat_05ms_20261017_162915  # one bag
python scripts/download_data.py --revision v0.1                  # pinned dataset version
python scripts/download_data.py --dry-run                        # list only, no download
```

Filters can be repeated and combined. Bags with `split=exclude`, such as `test` rides, are skipped unless you pass `--include-excluded` or name them with `--bag`. Bags that are already downloaded are skipped. Downloaded files are checked against the sha256 in the manifest (turn this off with `--no-verify`).

## Bag reports

[scripts/generate_bags_report.py](../scripts/generate_bags_report.py) creates an HTML report for each downloaded bag. Use the reports to check a bag before you train on it. Each report shows:

- general info and the bag's manifest row
- integrity checks: expected topics, gaps, latency
- per-topic rates
- speed and paths from each odometry source
- IMU, vibration and wheel plots
- RGB and depth previews

```bash
pip install rosbags numpy matplotlib opencv-python-headless pyyaml

python scripts/generate_bags_report.py                                  # all bags in data/raw/
python scripts/generate_bags_report.py --session 2026-10-17_ucu-cinema  # one session
python scripts/generate_bags_report.py --bag flat_05ms_20261017_162915  # one bag (name or path)
python scripts/generate_bags_report.py --skip-existing --jobs 4         # only new bags, in parallel
python scripts/generate_bags_report.py --index-only                     # only rebuild the index pages
```

Reports are written to `reports/`.

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
| `mixed`        | All terrain types: flat with bumps, single stones and passes through the stony area from different angles. |

## Bag naming

```
<ride_type>_<speed>[_<index>]_<YYYYMMDD>_<HHMMSS>
```

| Part          | Required | Meaning |
|---------------|----------|---------|
| `ride_type`   | yes      | One of the ride types above. |
| `speed`       | yes*     | Maximum speed in tenths of m/s, followed by `ms`: `03ms` = 0.3 m/s, `05ms` = 0.5 m/s, `08ms` = 0.8 m/s. This is the speed with the joystick fully forward. The actual speed may be lower. |
| `index`       | no       | Number of a repeated recording with the same ride type and speed in one session, starting from 1 (`_1`, `_2`, …). Used only when there is more than one such recording. |
| `YYYYMMDD`    | yes      | Recording date. |
| `HHMMSS`      | yes      | Recording start time (UTC+0). |

\* `test` bags have no speed, e.g. `test_20261017_162047`.

Examples:
- `flat_05ms_20261017_162915`: flat ride at max 0.5 m/s, recorded 2026-10-17 at 16:29:15
- `mixed_08ms_20261017_171002`: mixed ride at max 0.8 m/s
- `mixed_05ms_2_20261019_160254`: second mixed ride at max 0.5 m/s in the 2026-10-19 session
