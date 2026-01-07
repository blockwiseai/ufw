import argparse
import copy
import ipaddress
import json
import os
import subprocess
import sys
import time
from typing import Optional, Iterable, Set, Dict, Any, Tuple, List

import bittensor as bt
import numpy as np
import requests
import torch

try:
    import fcntl  # Linux only
except ImportError:
    fcntl = None


def log(message: str, level: str = "INFO") -> None:
    """Simple logging function that works well with PM2."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{timestamp} - {level} - {message}", flush=True)


def run_cmd(cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
        )
    return proc.returncode, proc.stdout, proc.stderr


def with_lock(lock_path: str):
    """Context manager for a simple file lock (prevents concurrent ufw modifications)."""

    class _Lock:
        def __init__(self, path: str):
            self.path = path
            self.f = None

        def __enter__(self):
            if fcntl is None:
                return self
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.f = open(self.path, "w")
            fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.f is not None:
                try:
                    fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
                finally:
                    self.f.close()
            return False

    return _Lock(lock_path)


def ensure_ssh_allowed() -> None:
    """Ensure an allow rule for SSH exists in UFW (idempotent)."""
    try:
        # Prefer the OpenSSH application profile (22/tcp).
        rc, _, _ = run_cmd(["sudo", "ufw", "allow", "OpenSSH"], check=False)
        if rc != 0:
            # Fallback if the OpenSSH profile is not available.
            run_cmd(["sudo", "ufw", "allow", "22/tcp"], check=False)

        log("Ensured SSH is allowed in UFW (OpenSSH / 22tcp)", "INFO")
    except Exception as e:
        log(f"Failed to ensure SSH is allowed via UFW: {e}", "ERROR")


def ensure_ufw_active() -> None:
    """Ensure UFW is enabled; do not disable/reset.

    Safety: if UFW is inactive, allow SSH before enabling to avoid lockout.
    """
    try:
        ensure_ssh_allowed()
        _, out, _ = run_cmd(["sudo", "ufw", "status"], check=False)
        if "Status: inactive" in out:
            log("UFW is inactive, enabling it", "WARNING")

            # Safety rule to avoid SSH lockout (idempotent).
            # If you don't want this, remove the next line.
            run_cmd(["sudo", "ufw", "allow", "OpenSSH"], check=False)

            run_cmd(["sudo", "ufw", "--force", "enable"], check=True)
            log("UFW enabled", "INFO")
    except Exception as e:
        log(f"Failed to check/enable UFW: {e}", "ERROR")


def normalize_to_cidrs(items: Iterable[str]) -> Set[str]:
    """
    Normalize inputs to valid IPv4 CIDR strings.
    Accepts either single IPv4 (converted to /32) or IPv4 CIDR.
    """
    out: Set[str] = set()
    for raw in items:
        s = (raw or "").strip()
        if not s:
            continue
        try:
            if "/" in s:
                net = ipaddress.ip_network(s, strict=False)
                if isinstance(net, ipaddress.IPv4Network):
                    out.add(str(net))
            else:
                ip = ipaddress.ip_address(s)
                if isinstance(ip, ipaddress.IPv4Address):
                    out.add(f"{ip}/32")
        except ValueError:
            log(f"Skipping invalid IP/CIDR: {s}", "WARNING")
    return out


def load_state(path: str) -> Dict[str, Any]:
    """Load state from disk (or return empty state)."""
    if not os.path.exists(path):
        return {"validators": [], "aws": []}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"validators": [], "aws": []}
        data.setdefault("validators", [])
        data.setdefault("aws", [])
        return data
    except Exception as e:
        log(f"Failed to read state file {path}: {e}", "WARNING")
        return {"validators": [], "aws": []}


def save_state(path: str, state: Dict[str, Any]) -> None:
    """Save state to disk atomically."""
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def ufw_allow_any_tcp_from_cidr(cidr: str, comment: str) -> None:
    """
    Add a UFW allow rule: allow any TCP from CIDR (no port restriction).

    Note: On some UFW versions, '--force' is NOT accepted for 'allow'.
    """
    base = ["sudo", "ufw", "allow", "from", cidr, "to", "any", "proto", "tcp"]
    cmd_with_comment = base + ["comment", comment]

    rc, _, _ = run_cmd(cmd_with_comment, check=False)
    if rc == 0:
        return

    # Retry without comment (covers versions without 'comment' support)
    rc2, _, err2 = run_cmd(base, check=False)
    if rc2 != 0:
        raise RuntimeError(
            f"Failed to add UFW rule: {' '.join(base)} -> {err2.strip()}"
        )


def ufw_delete_any_tcp_from_cidr(cidr: str) -> None:
    """Delete a UFW rule: allow any TCP from CIDR (non-interactive)."""
    cmd = [
        "sudo",
        "ufw",
        "--force",  # keep here, delete is interactive without it
        "delete",
        "allow",
        "from",
        cidr,
        "to",
        "any",
        "proto",
        "tcp",
    ]
    run_cmd(cmd, check=False)  # ignore if rule does not exist


def sync_ufw_cidr_rules(
    state_path: str,
    state_key: str,
    desired_cidrs: Set[str],
    comment: str,
    lock_path: str,
) -> None:
    """
    Apply delta updates to UFW without disabling/resetting.
    Only manages CIDRs tracked in the local state file.
    """
    with with_lock(lock_path):
        ensure_ufw_active()

        state = load_state(state_path)
        previous = set(state.get(state_key, []) or [])

        to_add = desired_cidrs - previous
        to_remove = previous - desired_cidrs

        if not to_add and not to_remove:
            log(
                f"No UFW changes needed for {state_key} (count={len(desired_cidrs)})",
                "INFO",
            )
            return

        log(
            f"Updating UFW rules for {state_key}: add={len(to_add)} remove={len(to_remove)}",
            "INFO",
        )

        # Remove old rules first (no downtime)
        for cidr in sorted(to_remove):
            try:
                ufw_delete_any_tcp_from_cidr(cidr)
            except Exception as e:
                log(f"Failed to remove rule for {cidr}: {e}", "WARNING")

        # Add new rules
        for cidr in sorted(to_add):
            try:
                ufw_allow_any_tcp_from_cidr(cidr, comment=comment)
            except Exception as e:
                log(f"Failed to add rule for {cidr}: {e}", "ERROR")

        state[state_key] = sorted(desired_cidrs)
        save_state(state_path, state)

        log(f"UFW update complete for {state_key} (now={len(desired_cidrs)})", "INFO")


def connect_to_subtensor(
    network: str = "finney", max_retries: int = 5
) -> Optional[bt.subtensor]:
    """Connect to subtensor with retries."""
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            log(f"Connecting to subtensor (attempt {attempt + 1}/{max_retries})")
            subtensor = bt.subtensor(network=network)
            log("Connected to subtensor")
            return subtensor
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                log(
                    f"Subtensor connect error: {e}. Retrying in {wait_time}s", "WARNING"
                )
                time.sleep(wait_time)
            else:
                log(
                    f"Failed to connect to subtensor after {max_retries} attempts: {e}",
                    "ERROR",
                )
                return None


def resync_metagraph(metagraph, subtensor, max_retries: int = 3) -> bool:
    """Resync metagraph with retries."""
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            log("Resyncing metagraph")
            previous_metagraph = copy.deepcopy(metagraph)
            metagraph.sync(subtensor=subtensor)
            if previous_metagraph.axons != metagraph.axons:
                log("Metagraph updated")
            log("Metagraph synced")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)
                log(f"Metagraph sync error: {e}. Retrying in {wait_time}s", "WARNING")
                time.sleep(wait_time)
            else:
                log(
                    f"Failed to sync metagraph after {max_retries} attempts: {e}",
                    "ERROR",
                )
                return False


def get_top_ips_from_metagraph(metagraph: bt.metagraph, top_n: int = 25) -> List[str]:
    """Return unique validator IPs for top stake validators."""
    start_time = time.time()
    log(f"Fetching top {top_n} IPs from metagraph")
    try:
        stakes = metagraph.stake
        non_zero_indices = np.where(stakes > 0)[0]
        if len(non_zero_indices) == 0:
            log("No validators with non-zero stake found", "WARNING")
            return []

        sorted_indices = non_zero_indices[np.argsort(stakes[non_zero_indices])]
        top_n = min(top_n, len(sorted_indices))
        top_stake_indices = sorted_indices[-top_n:]
        top_stake_indices = torch.tensor(
            top_stake_indices.tolist(), dtype=torch.long
        ).flip(0)

        top_uids = metagraph.uids[top_stake_indices].tolist()
        ips: List[str] = []
        for uid in top_uids:
            try:
                axon = metagraph.axons[uid]
                if axon and getattr(axon, "ip", None):
                    ips.append(axon.ip)
            except Exception as e:
                log(f"Error processing UID {uid}: {e}", "ERROR")

        unique_ips = list(dict.fromkeys(ips))
        log(f"Found {len(unique_ips)} unique IPs in {time.time() - start_time:.2f}s")
        return unique_ips
    except Exception as e:
        log(f"Error in get_top_ips_from_metagraph: {e}", "ERROR")
        return []


def fetch_aws_ipv4_cidrs(
    url: str = "https://ip-ranges.amazonaws.com/ip-ranges.json",
    timeout_s: int = 20,
    service: Optional[str] = None,
) -> Set[str]:
    """
    Fetch AWS IPv4 CIDRs from ip-ranges.json.

    If service is provided, filters by prefix['service'] == service.
    If service is None, returns all IPv4 prefixes.
    """
    if service:
        log(f"Fetching AWS IPv4 CIDRs (service filter: {service})", "INFO")
    else:
        log("Fetching AWS IPv4 CIDRs (no service filter: ALL AWS prefixes)", "INFO")

    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    cidrs: List[str] = []
    for prefix in data.get("prefixes", []):
        if service and prefix.get("service") != service:
            continue
        ip_prefix = prefix.get("ip_prefix")
        if ip_prefix:
            cidrs.append(ip_prefix)

    out = normalize_to_cidrs(cidrs)
    log(f"Fetched {len(out)} AWS IPv4 CIDRs", "INFO")
    return out


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="UFW Manager for Bittensor validators + optional AWS CIDRs"
    )

    parser.add_argument("--netuid", help="Netuid of the subnet", type=int, default=18)
    parser.add_argument(
        "--network", help="Bittensor network name", type=str, default="finney"
    )

    parser.add_argument(
        "--custom-ips",
        help="Comma-separated list of custom IPs/CIDRs to always allow (e.g. 1.2.3.4,10.0.0.0/8)",
        type=str,
        default="",
    )

    parser.add_argument(
        "--top-n", help="How many top validators to include", type=int, default=7
    )

    # AWS toggle
    parser.add_argument(
        "--disable-aws",
        help="Disable AWS CIDR fetching and rule management",
        action="store_true",
        default=False,
    )

    # Optional filter (default none = all AWS)
    parser.add_argument(
        "--aws-service",
        help="Optional AWS service filter from ip-ranges.json (default: none = all AWS prefixes)",
        type=str,
        default="",
    )

    parser.add_argument(
        "--aws-refresh-seconds",
        help="How often to refresh AWS CIDRs (default: 86400 = daily)",
        type=int,
        default=86400,
    )

    parser.add_argument(
        "--aws-url",
        help="AWS ip-ranges.json URL",
        type=str,
        default="https://ip-ranges.amazonaws.com/ip-ranges.json",
    )

    parser.add_argument(
        "--state-file",
        help="Path to store managed UFW state (so we can update deltas safely)",
        type=str,
        default="./state.json",
    )
    parser.add_argument(
        "--lock-file",
        help="Path to a lock file to prevent concurrent UFW updates",
        type=str,
        default="./ufw-manager.lock",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    use_aws = not args.disable_aws
    aws_service = args.aws_service.strip() or None

    custom_items = [ip.strip() for ip in args.custom_ips.split(",") if ip.strip()]
    custom_cidrs = normalize_to_cidrs(custom_items)
    if custom_cidrs:
        log(f"Custom allowlist entries: {sorted(custom_cidrs)}", "INFO")

    next_aws_refresh = 0.0

    try:
        log("Starting UFW Manager", "INFO")

        # AWS rules: once at startup, then daily (optional)
        if use_aws:
            try:
                aws_cidrs = fetch_aws_ipv4_cidrs(url=args.aws_url, service=aws_service)
                sync_ufw_cidr_rules(
                    state_path=args.state_file,
                    state_key="aws",
                    desired_cidrs=aws_cidrs,
                    comment="aws",
                    lock_path=args.lock_file,
                )
            except Exception as e:
                log(
                    f"AWS CIDR refresh failed at startup (keeping existing managed AWS rules): {e}",
                    "WARNING",
                )
            next_aws_refresh = time.time() + float(args.aws_refresh_seconds)
        else:
            log("AWS CIDR management disabled", "INFO")

        while True:
            subtensor = connect_to_subtensor(network=args.network)
            if not subtensor:
                log("Cannot connect to subtensor; retrying in 60s", "ERROR")
                time.sleep(60)
                continue

            try:
                log("Initializing metagraph...", "INFO")
                metagraph = subtensor.metagraph(netuid=args.netuid)
                log("Metagraph initialized", "INFO")

                while True:
                    cycle_start = time.time()

                    # Periodic AWS refresh (daily)
                    now = time.time()
                    if use_aws and now >= next_aws_refresh:
                        try:
                            aws_cidrs = fetch_aws_ipv4_cidrs(
                                url=args.aws_url, service=aws_service
                            )
                            sync_ufw_cidr_rules(
                                state_path=args.state_file,
                                state_key="aws",
                                desired_cidrs=aws_cidrs,
                                comment="aws",
                                lock_path=args.lock_file,
                            )
                        except Exception as e:
                            log(
                                f"AWS CIDR refresh failed (keeping existing managed AWS rules): {e}",
                                "WARNING",
                            )
                        next_aws_refresh = now + float(args.aws_refresh_seconds)

                    if not resync_metagraph(metagraph, subtensor):
                        log(
                            "Failed to sync metagraph; restarting connection cycle...",
                            "WARNING",
                        )
                        break

                    top_ips = get_top_ips_from_metagraph(metagraph, top_n=args.top_n)
                    top_cidrs = normalize_to_cidrs(top_ips)

                    desired_validators = set(top_cidrs) | set(custom_cidrs)

                    sync_ufw_cidr_rules(
                        state_path=args.state_file,
                        state_key="validators",
                        desired_cidrs=desired_validators,
                        comment="validators",
                        lock_path=args.lock_file,
                    )

                    log(
                        f"Complete update cycle took {time.time() - cycle_start:.2f}s",
                        "INFO",
                    )
                    log("Waiting 10 minutes before next update...", "INFO")
                    time.sleep(60 * 10)

            except Exception as e:
                log(f"Error in main loop: {e}", "ERROR")
                log("Restarting connection cycle in 60s...", "WARNING")
                time.sleep(60)

    except KeyboardInterrupt:
        log("Script terminated by user", "INFO")
        sys.exit(0)
    except Exception as e:
        log(f"Fatal error: {e}", "ERROR")
        sys.exit(1)
