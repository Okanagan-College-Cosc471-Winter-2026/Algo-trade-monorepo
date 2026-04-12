#! /usr/bin/env bash

set -e
set -x

python app/backend_pre_start.py

ALEMBIC_STATE=$(python - <<'PY'
from sqlalchemy import create_engine, text
from app.core.config import settings

engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))
with engine.connect() as conn:
    market_stocks = conn.execute(text("select to_regclass($$market.stocks$$) is not null")).scalar()
    alembic_table = conn.execute(text("select to_regclass($$public.alembic_version$$) is not null")).scalar()
    version_count = 0
    if alembic_table:
        version_count = conn.execute(text("select count(*) from public.alembic_version")).scalar()
    print(f"{int(bool(market_stocks))} {int(bool(alembic_table))} {int(version_count or 0)}")
PY
)

read -r MARKET_STOCKS_EXISTS ALEMBIC_TABLE_EXISTS ALEMBIC_VERSION_COUNT <<< "$ALEMBIC_STATE"

if [ "$MARKET_STOCKS_EXISTS" = "1" ] && [ "$ALEMBIC_VERSION_COUNT" = "0" ]; then
    alembic stamp head
else
    alembic upgrade head
fi

python app/initial_data.py
