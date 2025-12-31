import copy
import time
import argparse
import bittensor as bt
import subprocess
import numpy as np
import torch
import sys
from typing import Optional
import requests


def log(message, level="INFO"):
    """Simple logging function that works well with PM2"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"{timestamp} - {level} - {message}", flush=True)


def connect_to_subtensor(
    network: str = "finney", max_retries: int = 5
) -> Optional[bt.subtensor]:
    """
    Próbuje połączyć się z subtensorem z możliwością ponownych prób
    """
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            log(f"Próba połączenia z subtensorem (próba {attempt + 1}/{max_retries})")
            subtensor = bt.subtensor(network=network)
            log("Pomyślnie połączono z subtensorem")
            return subtensor
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)  # Exponential backoff
                log(
                    f"Błąd połączenia: {str(e)}. Ponowna próba za {wait_time} sekund...",
                    "WARNING",
                )
                time.sleep(wait_time)
            else:
                log(
                    f"Nie udało się połączyć z subtensorem po {max_retries} próbach: {str(e)}",
                    "ERROR",
                )
                return None


def resync_metagraph(metagraph, subtensor, max_retries: int = 3):
    """
    Synchronizuje metagraph z możliwością ponownych prób
    """
    retry_delay = 5
    for attempt in range(max_retries):
        try:
            log("Resyncing metagraph")
            previous_metagraph = copy.deepcopy(metagraph)
            metagraph.sync(subtensor=subtensor)
            if previous_metagraph.axons != metagraph.axons:
                log("Metagraph updated, re-syncing hotkeys")
            log("Metagraph synced")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)  # Exponential backoff
                log(
                    f"Błąd synchronizacji metagraph: {str(e)}. Ponowna próba za {wait_time} sekund...",
                    "WARNING",
                )
                time.sleep(wait_time)
            else:
                log(
                    f"Nie udało się zsynchronizować metagraph po {max_retries} próbach: {str(e)}",
                    "ERROR",
                )
                return False


def get_top_ips_from_metagraph(metagraph: bt.metagraph, top_n=25):
    start_time = time.time()
    log(f"Fetching top {top_n} IPs from metagraph")

    try:
        stakes = metagraph.stake
        log(
            f"Stakes stats - min: {stakes.min():.2f}, max: {stakes.max():.2f}, mean: {stakes.mean():.2f}"
        )

        non_zero_indices = np.where(stakes > 0)[0]
        log(f"Found {len(non_zero_indices)} validators with non-zero stake")

        if len(non_zero_indices) == 0:
            log("No validators with non-zero stake found", "WARNING")
            return []

        sorted_indices = non_zero_indices[np.argsort(stakes[non_zero_indices])]
        top_n = min(top_n, len(sorted_indices))
        top_stake_indices = sorted_indices[-top_n:]
        top_stake_indices = torch.tensor(top_stake_indices.tolist(), dtype=torch.long)
        top_stake_indices = top_stake_indices.flip(0)

        top_uids = metagraph.uids[top_stake_indices].tolist()
        ips = []
        for uid in top_uids:
            try:
                axon = metagraph.axons[uid]
                if axon and hasattr(axon, "ip") and axon.ip:
                    ips.append(axon.ip)
                else:
                    log(f"Invalid axon data for UID {uid}", "WARNING")
            except Exception as e:
                log(f"Error processing UID {uid}: {str(e)}", "ERROR")

        unique_ips = list(dict.fromkeys(ips))
        duration = time.time() - start_time
        log(f"Found {len(unique_ips)} unique IPs in {duration:.2f} seconds")
        return unique_ips

    except Exception as e:
        log(f"Error in get_top_ips_from_metagraph: {str(e)}", "ERROR")
        return []


def update_ufw_with_ips(new_ips, custom_ips):
    if not new_ips and not custom_ips:
        log("No IPs to whitelist, keeping existing UFW rules")
        return

    all_ips = list(set(new_ips + custom_ips))

    start_time = time.time()
    log("Starting UFW rules update")

    try:
        subprocess.run("sudo ufw disable", shell=True, check=True)

        result = subprocess.run(
            "sudo ufw status numbered", shell=True, capture_output=True, text=True
        )
        current_rules = result.stdout

        current_ips = []
        for line in current_rules.split("\n"):
            if "from" in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "from" and i + 1 < len(parts):
                        ip = parts[i + 1].split("/")[0]
                        if ip != "Anywhere" and ip != "0.0.0.0":
                            current_ips.append(ip)

        ips_to_add = set(all_ips) - set(current_ips)
        ips_to_remove = set(current_ips) - set(all_ips)

        if not ips_to_add and not ips_to_remove:
            log("No changes in IP list needed")
            return

        log(f"Need to add {len(ips_to_add)} IPs and remove {len(ips_to_remove)} IPs")

        subprocess.run("yes | sudo ufw reset", shell=True, check=True)
        subprocess.run("sudo ufw allow ssh", shell=True, check=True)
        log("Added SSH access rule")

        for ip in all_ips:
            if ip and ip != "0.0.0.0":
                cmd = f"sudo ufw allow proto tcp from {ip}/32"
                start = time.time()
                try:
                    subprocess.run(cmd, shell=True, check=True)
                    log(f"Added IP {ip} ({time.time() - start:.2f}s)")
                except subprocess.CalledProcessError as e:
                    log(f"Error adding IP {ip}: {e}", "ERROR")

        subprocess.run("yes | sudo ufw enable", shell=True, check=True)
        log("UFW enabled")

        result = subprocess.run(
            "sudo ufw status", shell=True, capture_output=True, text=True
        )
        if "Status: active" in result.stdout:
            log("UFW is active and configured")
        else:
            log("Warning: UFW might not be active", "WARNING")

        duration = time.time() - start_time
        log(f"Completed UFW update in {duration:.2f} seconds")

    except Exception as e:
        log(f"Error updating UFW rules: {e}", "ERROR")
        try:
            subprocess.run("sudo ufw allow ssh", shell=True)
            subprocess.run("yes | sudo ufw enable", shell=True)
        except:
            log("Failed to ensure SSH access after error", "ERROR")


def get_aws_healtcheck_endpoint(
    url: str = "https://ip-ranges.amazonaws.com/ip-ranges.json",
):
    response = requests.get(url)
    response_json = response.json()
    response_json.keys()
    prefixes = response_json.get("prefixes", None)
    healtcheck_endpoints = []
    if prefixes:
        for prefix in prefixes:
            ip_prefix = prefix.get("ip_prefix")
            if not ip_prefix:
                continue
            ip_clean = ip_prefix.split("/")[0]
            # print(ip_clean)
            healtcheck_endpoints.append(ip_clean)
    return healtcheck_endpoints


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run firewall script for Bittensor")
    parser.add_argument("--netuid", help="Netuid of the subnet", type=int, default=18)
    parser.add_argument(
        "--custom-ips",
        help="Comma-separated list of custom IPs to always include",
        type=str,
        default="",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Process custom IPs
    custom_ips = [ip.strip() for ip in args.custom_ips.split(",") if ip.strip()]
    if custom_ips:
        log(f"Added {len(custom_ips)} custom IPs to whitelist: {', '.join(custom_ips)}")
    healtcheck_endpoints = get_aws_healtcheck_endpoint()

    all_ips = custom_ips + healtcheck_endpoints
    try:
        log("Starting UFW Manager")

        while True:
            # Próbuj połączyć się z subtensorem
            subtensor = connect_to_subtensor()
            if not subtensor:
                log(
                    "Nie można połączyć się z subtensorem. Ponowna próba za 60 sekund...",
                    "ERROR",
                )
                time.sleep(60)
                continue

            try:
                # Inicjalizacja metagraph
                log("Initializing metagraph...")
                metagraph = subtensor.metagraph(netuid=args.netuid)
                log("Successfully initialized metagraph")

                while True:
                    cycle_start = time.time()

                    # Resync metagraph z obsługą ponownych prób
                    if not resync_metagraph(metagraph, subtensor):
                        # Jeśli synchronizacja nie powiodła się, zaczynamy od nowa
                        log("Failed to sync metagraph, restarting connection cycle...")
                        break

                    # Get top IPs
                    top_ips = get_top_ips_from_metagraph(metagraph, top_n=7)
                    log(top_ips)
                    # Update UFW with both metagraph and custom IPs
                    update_ufw_with_ips(top_ips, custom_ips)

                    cycle_duration = time.time() - cycle_start
                    log(f"Complete update cycle took {cycle_duration:.2f} seconds")

                    log("Waiting for 10mins before next update...")
                    time.sleep(60 * 10)

            except Exception as e:
                log(f"Error in main loop: {e}", "ERROR")
                log("Restarting connection cycle...")
                time.sleep(60)

    except KeyboardInterrupt:
        log("Script terminated by user")
    except Exception as e:
        log(f"Fatal error: {e}", "ERROR")
