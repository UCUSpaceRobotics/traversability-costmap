# Network helpers for scripts that talk to the rover. Source this file, do not run it.

# ===========================================================================
# FUNCTION: ensure_wifi_connection
# PURPOSE:  Connects to the target SSID unless already connected. Supports Linux (nmcli) and macOS.
# USAGE:    ensure_wifi_connection "SSID" ["PASSWORD"] ["USE_ETH (true/false)"]
# ===========================================================================
ensure_wifi_connection() {
  local target_ssid="${1:-}"
  local password="${2:-}"
  local use_eth="${3:-false}"

  if [ "$use_eth" = "true" ]; then
    echo -e "\e[34m[INFO]\e[0m Ethernet mode active. Skipping Wi-Fi connection."
    return 0
  fi
  if [ -z "$target_ssid" ]; then
    echo -e "\e[33m[WARNING]\e[0m No SSID provided. Skipping Wi-Fi auto-connect."
    return 0
  fi

  if command -v nmcli >/dev/null 2>&1; then
    local current_ssid
    current_ssid=$(nmcli -t -f active,ssid dev wifi 2>/dev/null | grep '^yes' | cut -d: -f2 || true)
    if [ "$current_ssid" = "$target_ssid" ]; then
      echo -e "\e[32m[INFO]\e[0m Already connected to Wi-Fi: ${target_ssid}"
      return 0
    fi
    echo -e "\e[34m[INFO]\e[0m Attempting to connect to Wi-Fi: ${target_ssid}..."
    if [ -n "$password" ]; then
      nmcli device wifi connect "$target_ssid" password "$password" >/dev/null 2>&1 || true
    else
      nmcli device wifi connect "$target_ssid" >/dev/null 2>&1 || true
    fi

  elif command -v networksetup >/dev/null 2>&1; then
    local wifi_iface current_ssid
    wifi_iface=$(networksetup -listallhardwareports 2>/dev/null | awk '/Hardware Port: Wi-Fi/{getline; print $2}' || true)
    if [ -z "$wifi_iface" ]; then
      echo -e "\e[31m[ERROR]\e[0m Could not detect a Wi-Fi interface on this Mac."
      return 1
    fi
    current_ssid=$(networksetup -getairportnetwork "$wifi_iface" 2>/dev/null | awk -F': ' '{print $2}' || true)
    if [ "$current_ssid" = "$target_ssid" ]; then
      echo -e "\e[32m[INFO]\e[0m Already connected to Wi-Fi: ${target_ssid}"
      return 0
    fi
    echo -e "\e[34m[INFO]\e[0m Attempting to connect to Wi-Fi: ${target_ssid}..."
    if [ -n "$password" ]; then
      networksetup -setairportnetwork "$wifi_iface" "$target_ssid" "$password" >/dev/null 2>&1 || true
    else
      networksetup -setairportnetwork "$wifi_iface" "$target_ssid" >/dev/null 2>&1 || true
    fi

  else
    echo -e "\e[33m[WARNING]\e[0m OS not supported for auto-connect. Please switch to '${target_ssid}' manually."
  fi
}

# ===========================================================================
# FUNCTION: wait_for_ssh
# PURPOSE:  Polls a target over SSH until it responds; returns 1 on timeout.
# USAGE:    wait_for_ssh "user@host" [max_retries] [use_eth (true/false)]
# ===========================================================================
wait_for_ssh() {
  local target="$1"
  local max_retries="${2:-30}"
  local use_eth="${3:-false}"
  local retry_count=0

  if [ "$use_eth" = "true" ]; then
    echo -n "Waiting for SSH to ${target} over Ethernet... "
  else
    echo -n "Waiting for SSH to ${target} (connect to Wi-Fi manually if needed)... "
  fi

  while ! ssh -q -o BatchMode=yes -o ConnectTimeout=2 -o StrictHostKeyChecking=accept-new "${target}" "echo ok" > /dev/null 2>&1; do
    sleep 2
    echo -n "."
    retry_count=$((retry_count + 1))
    if [ "$retry_count" -ge "$max_retries" ]; then
      echo ""
      echo -e "\e[31m[ERROR]\e[0m Timeout: Could not connect to ${target}."
      return 1
    fi
  done

  echo ""
  echo -e "\e[32m[SUCCESS]\e[0m Connection established."
}
