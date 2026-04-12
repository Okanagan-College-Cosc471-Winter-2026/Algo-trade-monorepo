#!/bin/bash
set -e
printenv | grep -v "no_proxy" > /etc/environment
mkdir -p /app/logs
exec cron -f
