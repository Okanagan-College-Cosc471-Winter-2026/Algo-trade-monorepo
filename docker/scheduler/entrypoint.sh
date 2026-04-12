#!/bin/bash
set -e
printenv | grep -v "no_proxy" > /etc/environment
mkdir -p /app/logs

# Set up SSH key for NIBI if provided
if [ -n "${NIBI_SSH_KEY_CONTENT}" ]; then
    mkdir -p /root/.ssh
    echo "${NIBI_SSH_KEY_CONTENT}" > /root/.ssh/nibi_key
    chmod 600 /root/.ssh/nibi_key
    echo "StrictHostKeyChecking no" >> /root/.ssh/config
    export NIBI_SSH_KEY=/root/.ssh/nibi_key
fi

exec cron -f
