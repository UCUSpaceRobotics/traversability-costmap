# Data

ROS 2 bags (`.mcap`) recorded on the Indomitus rover while driving over different terrain. Each bag contains ZED 2i RGB-D camera streams together with the rover's IMU and odometry. These signals serve as self-supervised labels (slip and chassis vibration) for training the traversability model.

Recordings are grouped into **sessions**: one session is one recording day at one location, and it has a single camera configuration.

Dataset on Hugging Face: [ucu-space-robotics/traversability-costmap](https://huggingface.co/datasets/ucu-space-robotics/traversability-costmap)

## Contents

- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Download](#download)
    - [Without the script](#without-the-script)
- [Bag reports](#bag-reports)
- [Recording](#recording)
- [Adding a session](#adding-a-session)
- [Directory structure](#directory-structure)
  - [Hugging Face repo](#hugging-face-repo)
- [Ride types](#ride-types)
- [Bag naming](#bag-naming)


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

Bags are not stored in git. Use [scripts/sync_data.py](../scripts/sync_data.py) to fetch them into `data/raw/`:

```bash
python scripts/sync_data.py download                                  # all train + val bags
python scripts/sync_data.py download --split val                      # only validation bags
python scripts/sync_data.py download --session 2026-10-17_ucu-cinema  # one session
python scripts/sync_data.py download --ride-type mixed                # one ride type
python scripts/sync_data.py download --bag 20261017_162915_v0p5  # one bag
python scripts/sync_data.py download --revision v0.1                  # pinned dataset version
python scripts/sync_data.py download --dry-run                        # list only, no download
```

Filters can be repeated and combined. Bags with `split=exclude`, such as `test` rides, are skipped unless you pass `--include-excluded` or name them with `--bag`. Bags that are already downloaded are skipped.

#### Without the script

Use the `hf` CLI directly. The Hub repo has the same layout as `data/` (see [Hugging Face repo](#hugging-face-repo)), so download into `data/` with `--include "*.mcap"`. Without that filter, the Hub copies of the YAML files overwrite the newer ones from git.

```bash
hf download ucu-space-robotics/traversability-costmap --repo-type dataset --local-dir data --include "*.mcap"                                     # all bags
hf download ucu-space-robotics/traversability-costmap --repo-type dataset --local-dir data --include "raw/2026-10-19_ucu-cinema/*.mcap"           # one session
hf download ucu-space-robotics/traversability-costmap --repo-type dataset --local-dir data --include "raw/*/20261019_152249_v0p5/*.mcap"   # one bag
```

## Bag reports

[scripts/generate_bags_report.py](../scripts/generate_bags_report.py) creates an HTML report for each downloaded bag. Use the reports to check a bag before you train on it. Each report shows:

- general info and the bag's manifest row
- integrity checks: expected topics, gaps, latency
- per-topic rates
- speed and paths from each odometry source
- IMU, vibration and wheel plots
- RGB and depth previews

```bash
python scripts/generate_bags_report.py                                  # all bags in data/raw/
python scripts/generate_bags_report.py --session 2026-10-17_ucu-cinema  # one session
python scripts/generate_bags_report.py --bag 20261017_162915_v0p5  # one bag (name or path)
python scripts/generate_bags_report.py --skip-existing --jobs 4         # only new bags, in parallel
python scripts/generate_bags_report.py --index-only                     # only rebuild the index pages
```

Reports are written to `reports/` which are ignored by the `.gitignore`.

## Recording

Use [scripts/record_bag.sh](../scripts/record_bag.sh) on the laptop to record new bags on the rover. The script connects to the rover over the hotspot (`--eth` for cable, `--no-wifi` if already connected), and runs [scripts/rover/bag_recorder.py](../scripts/rover/bag_recorder.py) inside the `rover_prod` container. The rover stack must be running. Then the script:

1. checks every topic: it exists, has the right type, has publishers and its rate is close to the expected one;
2. records until Ctrl+C or `--duration`, and checks the topics again every few seconds, with a warning for each problem;
3. prints a summary that compares each topic's message count with rate × duration;
4. with `--copy-to`, copies the bag to the laptop.

The bag name follows the [naming convention](#bag-naming). The topics, expected rates, tolerances and recording settings are in [config/recording.yaml](../config/recording.yaml).

```bash
scripts/record_bag.sh check                                                  # pre-flight checks only
scripts/record_bag.sh record --speed 0.5                                      # until Ctrl+C
scripts/record_bag.sh record --speed 0.8 --duration 5m \
    --copy-to data/raw/2026-10-20_ucu-cinema                                 # stop after 5 min, copy to the laptop
scripts/record_bag.sh list                                                   # bags on the rover
scripts/record_bag.sh summary 20261020_101500_v0p5                          # summary of a bag on the rover
scripts/record_bag.sh fetch --copy-to data/raw/2026-10-20_ucu-cinema         # copy the newest bag
scripts/record_bag.sh fetch 20261020_100000_v0p5 20261020_101500_v0p5 --copy-to data/raw/2026-10-20_ucu-cinema  # several bags
scripts/record_bag.sh fetch --from-file bags.txt --copy-to data/raw/2026-10-20_ucu-cinema  # bags listed in a file, one per line
scripts/record_bag.sh fetch --all --list --copy-to data/raw/2026-10-20_ucu-cinema  # show what --all would copy
scripts/record_bag.sh fetch --all --copy-to data/raw/2026-10-20_ucu-cinema   # copy all bags not copied yet
scripts/record_bag.sh stop                                                   # stop a recording left running after an ssh drop
scripts/record_bag.sh --help                                                 # all options
```

If the ssh connection drops during a recording, reconnect and run `scripts/record_bag.sh stop`, or wait for `--duration` to end. Either way the bag is finalised. Then check it with `summary`.

## Adding a session

You need a Hugging Face token with write access to `ucu-space-robotics` (`hf auth login`).

**1. Create the session folder.** Name it `<YYYY-MM-DD>_<location>`, record the bags into it with `--copy-to data/raw/<session>`, and add the two YAML files:

```
data/raw/2026-10-20_ucu-cinema/
├── session.yaml
├── zed2i_record.yaml             # ZED 2i config the rover used
└── <bag>/                        # one folder per bag, as copied by record_bag.sh
    ├── metadata.yaml
    └── <bag>_0.mcap
```

Minimal `session.yaml`:

```yaml
date: 2026-10-20
location: UCU Cinema
environment: indoor              # indoor / outdoor
lighting: artificial, constant
surface: flat compacted soil floor with small bumps, scattered single stones and one stony area
rover_code_commit: https://github.com/UCUSpaceRobotics/indomitus-rover-core/commit/<commit-hash>
zed_config: zed2i_record.yaml
notes: |
  - Anything unusual about the recording: gaps, broken topics, sensors that misbehaved.
```

**2. Check the bags.** Generate the reports and look for gaps, low rates or broken odometry. Write any problems in `notes`.

```bash
python scripts/generate_bags_report.py --session 2026-10-20_ucu-cinema
```

**3. Add the bags to [manifest.csv](manifest.csv),** one row per bag:

| Column | Value |
|---|---|
| `bag`, `session` | bag folder name and session folder name |
| `ride_type` | one of the [ride types](#ride-types) |
| `terrain` | `;`-separated terrain in the ride: `ground`, `bumps`, `stones`, `stony_area` |
| `max_speed_ms` | max speed in m/s, e.g. `0.5`; empty for `test` bags |
| `duration_s` | duration in seconds, rounded (`ros2 bag info <bag>` or `duration` in `metadata.yaml`, in ns) |
| `size_mb` | size of the `.mcap` in MB (`du -m <bag>/*.mcap`) |
| `split` | `train`, `val` or `exclude` (`test` bags and bags you should not train on) |
| `notes` | optional, short |

**4. Update the docs if needed.** For a new ride type, add it to [Ride types](#ride-types). If the recorded topics changed, also update the topic table in [hf_README.md](hf_README.md) and upload it (see [Hugging Face repo](#hugging-face-repo)).

**5. Upload the session to Hugging Face.** This uploads the bags and the YAML files in one commit:

```bash
hf upload ucu-space-robotics/traversability-costmap data/raw/2026-10-20_ucu-cinema raw/2026-10-20_ucu-cinema \
    --repo-type dataset --commit-message "add session 2026-10-20_ucu-cinema"
```

If the upload is interrupted, run the same command again. Files that are already uploaded are skipped.

**6. Commit the metadata to git.** The `.mcap` files are ignored by git, so this adds only the YAML files and the manifest:

```bash
git add data/raw/2026-10-20_ucu-cinema data/manifest.csv
git commit -m "chore(data): add session 2026-10-20_ucu-cinema"
```

## Directory structure

```
data/
├── README.md
├── hf_README.md                  # dataset card of the Hugging Face repo (uploaded there as README.md)
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

### Hugging Face repo

The [dataset repo](https://huggingface.co/datasets/ucu-space-robotics/traversability-costmap) has the same layout as `data/`, without `manifest.csv`:

```
ucu-space-robotics/traversability-costmap/
├── README.md                     # dataset card, from data/hf_README.md
├── .gitattributes                # stores .mcap files in LFS
└── raw/
    └── <session>/                # same as data/raw/<session>/
        ├── session.yaml
        ├── zed2i_record.yaml
        └── <bag>/
            ├── metadata.yaml
            └── <bag>_0.mcap
```

The YAML files on the Hub are copies of the ones in git, and they can be out of date. The git versions are the up-to-date ones, so [scripts/sync_data.py](../scripts/sync_data.py) downloads only the `.mcap` files.

The dataset card is kept in [hf_README.md](hf_README.md). After you change it, upload it:

```bash
hf upload ucu-space-robotics/traversability-costmap data/hf_README.md README.md --repo-type dataset
```

## Ride types

The ride type is not part of the bag name, because it is not always known when the recording starts. It is set after the ride, in the `ride_type` column of [manifest.csv](manifest.csv).

| Ride type      | Description |
|----------------|-------------|
| `test`         | Setup check. Never used for training or validation. |
| `flat`         | Mostly flat surface with small bumps and stones. |
| `singlestones` | Flat surface. The rover deliberately drives over and next to single stones. |
| `mixed`        | All terrain types: flat with bumps, single stones and passes through the stony area from different angles. |

## Bag naming

```
<YYYYMMDD>_<HHMMSS>[_v<speed>]
```

| Part       | Required | Meaning |
|------------|----------|---------|
| `YYYYMMDD` | yes      | Recording date (UTC+0). |
| `HHMMSS`   | yes      | Recording start time (UTC+0). |
| `v<speed>` | yes*     | Maximum speed in m/s, with `p` instead of the decimal point: `v0p3` = 0.3 m/s, `v0p5` = 0.5 m/s, `v0p8` = 0.8 m/s, `v1p0` = 1.0 m/s. This is the speed limit set for the ride (joystick fully forward). The actual speed may be lower. |

\* Setup test bags are recorded without `--speed` and have no speed part, e.g. `20261017_162047`.

Bags sort by recording time. The date and time make the name unique, so there is no index. The ride type, terrain and the rest are in [manifest.csv](manifest.csv).

Examples:
- `20261017_162915_v0p5`: recorded 2026-10-17 at 16:29:15 with the max speed set to 0.5 m/s
- `20261115_125625_v1p0`: recorded 2026-11-15 at 12:56:25 with the max speed set to 1.0 m/s
- `20261017_162047`: setup test, no speed
