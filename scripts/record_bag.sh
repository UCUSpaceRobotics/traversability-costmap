#!/usr/bin/env bash
# Record dataset bags on the rover from the laptop. Run with --help for usage.
#
# The work is done by scripts/rover/bag_recorder.py inside the rover container; this script connects
# to the rover, copies the recorder and its config into the container, runs it in an interactive
# terminal (so Ctrl+C reaches it) and copies bags back to the laptop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RECORDER="${SCRIPT_DIR}/rover/bag_recorder.py"
REMOTE_TMP="/tmp/bag_recorder"   # inside the container

source "${SCRIPT_DIR}/lib/net.sh"

# DEFAULTS (flags override them; environment variables set the defaults)
JETSON_USER="${ROVER_USER:-indomitus-rover}"
JETSON_HOTSPOT_IP="10.42.0.1"
JETSON_ETHERNET_IP="indomitus-rover-computer.local"
JETSON_IP="${ROVER_IP:-$JETSON_HOTSPOT_IP}"
CONTAINER_NAME="${ROVER_CONTAINER:-}"      # default depends on --local, set after parsing
ROS_DISTRO_NAME="${ROVER_ROS_DISTRO:-humble}"
WORKSPACE_DIR="${ROVER_WORKSPACE:-/opt/ws}"
WIFI_SSID="${ROVER_WIFI_SSID:-ERC_UCUSpaceRobotics_A}"
WIFI_PASS="${ROVER_WIFI_PASS:-19283746}"
CONFIG="${REPO_ROOT}/config/recording.yaml"
COPY_TO=""
USE_ETH=false
NO_WIFI=false
LOCAL=false

success() { echo -e "\e[32m[SUCCESS]\e[0m $1"; }
warning() { echo -e "\e[33m[WARNING]\e[0m $1"; }
error()   { echo -e "\e[31m[ERROR]\e[0m $1" >&2; exit 1; }
step()    { echo -e "\n\e[34m>>> $1\e[0m"; }

show_help() {
  cat << EOF
Usage: $0 COMMAND [OPTIONS] [RECORDER OPTIONS]

Records ROS 2 bags inside the rover container from the laptop. Topics, expected rates and
recording settings are in config/recording.yaml.

Commands:
  record        Check topics and publishers, record until Ctrl+C or --duration, monitor topics
                while recording, then print a summary (message counts vs. expected).
  check         Only run the pre-flight checks.
  summary [BAG] Print the summary of a bag on the rover (default: newest).
  list          List the bags on the rover.
  stop          Stop a recording that is still running (e.g. after the ssh connection dropped).
  fetch [BAG]   Copy a bag from the rover to --copy-to (default: newest bag).

Options:
  --copy-to DIR       record/fetch: copy the bag to DIR on this computer (DIR/<bag>/)
  --config FILE       Recording config (Default: config/recording.yaml)
  --local             Use a container on this computer instead of the rover (e.g. for testing
                      with 'ros2 bag play'); default container: rover_dev
  --eth               Use wired Ethernet connection (${JETSON_ETHERNET_IP}) instead of hotspot.
  --no-wifi, -n       Skip the Wi-Fi auto-connect and ssh straight to the Jetson.
  --user USER         Jetson SSH username (Default: ${JETSON_USER}, env ROVER_USER)
  --ip IP             Jetson IP address (Default: ${JETSON_IP}, env ROVER_IP)
  --container NAME    Docker container name (Default: rover_prod, env ROVER_CONTAINER)
  --ros-distro DIST   ROS 2 distribution in the container (Default: ${ROS_DISTRO_NAME})
  --workspace DIR     ROS 2 workspace in the container (Default: ${WORKSPACE_DIR})
  --ssid SSID         Wi-Fi SSID of the Jetson hotspot (Default: ${WIFI_SSID})
  --pass PASS         Wi-Fi password of the hotspot (Default: env ROVER_WIFI_PASS)
  -h, --help          Show this help; with a command also the recorder options

All other options are passed to the recorder, see '$0 record --help'.

Examples:
  $0 check
  $0 record --ride-type mixed --speed 0.5
  $0 record --ride-type flat --speed 0.8 --index 2 --duration 5m --copy-to data/raw/2026-10-20_ucu-cinema
  $0 fetch mixed_05ms_20261020_101500 --copy-to data/raw/2026-10-20_ucu-cinema
EOF
}

need_arg() { [[ "$2" -ge 2 ]] || error "$1 requires an argument."; }

# ==========================================
# ARGUMENT PARSING
# ==========================================

[[ "$#" -gt 0 ]] || { show_help; exit 1; }
COMMAND="$1"; shift
case "$COMMAND" in
  record|check|summary|list|stop|fetch) ;;
  -h|--help) show_help; exit 0;;
  *) error "Unknown command: '${COMMAND}'.\nRun '$0 --help' for usage.";;
esac

RECORDER_ARGS=()   # forwarded to bag_recorder.py
OUT_DIR_ARGS=()    # --out-dir is forwarded and also needed to find the bag for copying
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --copy-to)    need_arg "$1" "$#"; COPY_TO="$2"; shift 2;;
    --config)     need_arg "$1" "$#"; CONFIG="$2"; shift 2;;
    --local)      LOCAL=true; shift;;
    --eth)        USE_ETH=true; JETSON_IP="${JETSON_ETHERNET_IP}"; shift;;
    --no-wifi|-n) NO_WIFI=true; shift;;
    --user)       need_arg "$1" "$#"; JETSON_USER="$2"; shift 2;;
    --ip)         need_arg "$1" "$#"; JETSON_IP="$2"; shift 2;;
    --container)  need_arg "$1" "$#"; CONTAINER_NAME="$2"; shift 2;;
    --ros-distro) need_arg "$1" "$#"; ROS_DISTRO_NAME="$2"; shift 2;;
    --workspace)  need_arg "$1" "$#"; WORKSPACE_DIR="$2"; shift 2;;
    --ssid)       need_arg "$1" "$#"; WIFI_SSID="$2"; shift 2;;
    --pass)       need_arg "$1" "$#"; WIFI_PASS="$2"; shift 2;;
    --out-dir)    need_arg "$1" "$#"; OUT_DIR_ARGS=(--out-dir "$2"); shift 2;;
    --out-dir=*)  OUT_DIR_ARGS=(--out-dir "${1#*=}"); shift;;
    -h|--help)
      show_help
      if [[ "$COMMAND" != fetch ]] && command -v python3 > /dev/null; then
        echo -e "\nRecorder options ('${COMMAND}'):"
        python3 "$RECORDER" "$COMMAND" --help | sed -n -e '/^options:/,$p' -e '/^optional arguments:/,$p'
      fi
      exit 0;;
    *) RECORDER_ARGS+=("$1"); shift;;
  esac
done

[[ -f "$CONFIG" ]] || error "Config not found: ${CONFIG}"
[[ -z "$COPY_TO" || "$COMMAND" == record || "$COMMAND" == fetch ]] || error "--copy-to works only with record and fetch."
[[ -n "$COPY_TO" || "$COMMAND" != fetch ]] || error "fetch requires --copy-to DIR."
CONTAINER_NAME="${CONTAINER_NAME:-$($LOCAL && echo rover_dev || echo rover_prod)}"
TARGET="${JETSON_USER}@${JETSON_IP}"

# Interactive terminal only if we have one (Ctrl+C is then forwarded to the recorder)
if [[ -t 0 && -t 1 ]]; then DOCKER_TTY="-it"; else DOCKER_TTY="-i"; fi

SSH_OPTS=(
  -o ControlMaster=auto
  -o "ControlPath=/tmp/ssh_mux_%h_%p_%r"
  -o ControlPersist=10m
  -o StrictHostKeyChecking=accept-new
)

cleanup() {
  $LOCAL || ssh -q -O exit "${SSH_OPTS[@]}" "${TARGET}" 2>/dev/null || true
}
trap cleanup EXIT

# ==========================================
# HELPERS
# ==========================================

# Run a shell command string on the rover host (or on this computer with --local).
host_run() {
  if $LOCAL; then bash -c "$1"; else ssh -q "${SSH_OPTS[@]}" "${TARGET}" "$1"; fi
}

# Same, with a terminal when we have one.
host_run_tty() {
  if $LOCAL; then bash -c "$1"
  elif [[ "$DOCKER_TTY" == "-it" ]]; then ssh -tq "${SSH_OPTS[@]}" "${TARGET}" "$1"
  else ssh -q "${SSH_OPTS[@]}" "${TARGET}" "$1"
  fi
}

# Print the command string that runs the recorder in the container with ROS sourced.
# Usage: recorder_cmd DOCKER_EXEC_FLAGS RECORDER_ARGS...
recorder_cmd() {
  local flags="$1"; shift
  local ws_setup="${WORKSPACE_DIR}/install/setup.bash"
  local inner
  inner="source $(printf '%q' "/opt/ros/${ROS_DISTRO_NAME}/setup.bash")"
  inner+=" && { [ ! -f $(printf '%q' "$ws_setup") ] || source $(printf '%q' "$ws_setup"); }"
  inner+=" && exec python3 ${REMOTE_TMP}/bag_recorder.py $(printf '%q ' "$@")"
  printf 'docker exec %s %q bash -c %q' "$flags" "$CONTAINER_NAME" "$inner"
}

connect() {
  if $LOCAL; then
    step "Using local container '${CONTAINER_NAME}'..."
    docker info > /dev/null 2>&1 || error "Docker is not running."
  else
    step "Connecting to the rover (${TARGET})..."
    if $NO_WIFI; then
      echo "--no-wifi: skipping Wi-Fi auto-connect, ssh'ing directly"
    else
      ensure_wifi_connection "$WIFI_SSID" "$WIFI_PASS" "$USE_ETH"
    fi
    wait_for_ssh "$TARGET" 30 "$USE_ETH" || exit 1
  fi

  local state
  state=$(host_run "docker inspect -f '{{.State.Running}}' $(printf '%q' "$CONTAINER_NAME") 2>/dev/null" || true)
  [[ "$state" == "true" ]] || error "Container '${CONTAINER_NAME}' is not running. Start the rover stack first (e.g. enter_container.sh rover)."

  # The recorder always runs the laptop's version of the script and config
  local exec_prefix="docker exec -i $(printf '%q' "$CONTAINER_NAME") sh -c"
  host_run "${exec_prefix} 'mkdir -p ${REMOTE_TMP} && cat > ${REMOTE_TMP}/bag_recorder.py'" < "$RECORDER"
  host_run "${exec_prefix} 'cat > ${REMOTE_TMP}/recording.yaml'" < "$CONFIG"
}

# Copy a bag from the container to COPY_TO. Usage: fetch_bag [BAG]
# Uses rsync through 'docker exec' when the container has it (resumable), otherwise a tar stream.
fetch_bag() {
  local remote_path name parent size
  remote_path=$(host_run "$(recorder_cmd -i path "$@" ${OUT_DIR_ARGS[@]+"${OUT_DIR_ARGS[@]}"})") \
    || error "Could not find the bag on the rover."
  name="$(basename "$remote_path")"
  parent="$(dirname "$remote_path")"
  mkdir -p "$COPY_TO"
  size=$(host_run "docker exec $(printf '%q' "$CONTAINER_NAME") du -sb $(printf '%q' "$remote_path")" | cut -f1 || true)
  size="${size:-0}"
  step "Copying ${name} ($(( size / 1024 / 1024 )) MB) to ${COPY_TO}/..."

  if ! $LOCAL && command -v rsync > /dev/null \
      && host_run "docker exec $(printf '%q' "$CONTAINER_NAME") sh -c 'command -v rsync'" > /dev/null 2>&1; then
    rsync -a --partial --info=progress2 \
      -e "ssh -q ${SSH_OPTS[*]}" \
      --rsync-path="docker exec -i ${CONTAINER_NAME} rsync" \
      "${TARGET}:${remote_path}" "${COPY_TO}/"
  else
    local tar_cmd
    tar_cmd="docker exec $(printf '%q' "$CONTAINER_NAME") tar -C $(printf '%q' "$parent") -cf - $(printf '%q' "$name")"
    if command -v pv > /dev/null; then
      host_run "$tar_cmd" | pv -s "$size" | tar -xf - -C "$COPY_TO"
    else
      host_run "$tar_cmd" | tar -xf - -C "$COPY_TO"
    fi
  fi
  success "Bag copied to ${COPY_TO}/${name}"
}

# ==========================================
# MAIN
# ==========================================

connect

if [[ "$COMMAND" == fetch ]]; then
  fetch_bag ${RECORDER_ARGS[@]+"${RECORDER_ARGS[@]}"}
  exit 0
fi

rc=0
host_run_tty "$(recorder_cmd "$DOCKER_TTY" "$COMMAND" \
  ${RECORDER_ARGS[@]+"${RECORDER_ARGS[@]}"} ${OUT_DIR_ARGS[@]+"${OUT_DIR_ARGS[@]}"})" || rc=$?

if [[ "$COMMAND" == record && -n "$COPY_TO" ]]; then
  # 0 = recorded, 2 = recorded with a summary that shows problems; otherwise nothing was recorded
  if [[ "$rc" -eq 0 || "$rc" -eq 2 ]]; then
    fetch_bag
  else
    warning "No bag was recorded, nothing to copy."
  fi
fi
exit "$rc"
