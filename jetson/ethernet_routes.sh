#!/usr/bin/env bash
# Optional: prefer wired Ethernet when the Orin is on Wi-Fi and Ethernet in the same subnet.
# On our unit NetworkManager activated end0 but did not install its routes, so everything kept
# going over Wi-Fi (~2 MB/s laptop<->Jetson vs ~50 MB/s wired). Not persistent across reboots.
#
#   sudo IFACE=end0 GW=192.168.1.1 ./ethernet_routes.sh
set -euo pipefail
IFACE=${IFACE:-end0}
GW=${GW:-192.168.1.1}
ADDR=$(ip -4 -o addr show "$IFACE" | awk '{print $4}' | head -1)
[ -n "$ADDR" ] || { echo "$IFACE has no IPv4 address"; exit 1; }
IP=${ADDR%/*}
NET=$(python3 -c "import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)" "$ADDR")
ip route replace "$NET" dev "$IFACE" src "$IP" metric 100
ip route replace default via "$GW" dev "$IFACE" src "$IP" metric 100
ip route | grep -E "default|$NET"
