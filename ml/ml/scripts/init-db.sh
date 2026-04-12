#!/bin/bash
set -e

BACKUP_FILE="/docker-entrypoint-initdb.d/backup/market_schema.backup"
RESTORE_MARKER="${PGDATA:-/var/lib/postgresql/data/pgdata}/.restore_complete"

if [ -f "$BACKUP_FILE" ]; then
    echo "Restoring database from backup using pg_restore..."

    pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner --no-privileges -v "$BACKUP_FILE" 2>&1 || {
        echo "Warning: pg_restore completed with some warnings (this is often normal)"
    }

    echo "Database restore completed!"
else
    echo "No backup file found at $BACKUP_FILE, starting with empty database."
fi

touch "$RESTORE_MARKER"
echo "Restore marker written to $RESTORE_MARKER"
