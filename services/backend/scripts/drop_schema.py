from sqlalchemy import text

from app.core.db import engine


def drop_market_schema():
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS market CASCADE"))
        conn.commit()
    print("Dropped market schema.")


if __name__ == "__main__":
    drop_market_schema()
