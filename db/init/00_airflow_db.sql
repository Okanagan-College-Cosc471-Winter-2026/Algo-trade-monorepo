-- Create the Airflow metadata database.
-- Runs once at Postgres container initialisation (docker-entrypoint-initdb.d).
-- Idempotent: no-op if the database already exists.
SELECT 'CREATE DATABASE airflow OWNER appuser'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')
\gexec
