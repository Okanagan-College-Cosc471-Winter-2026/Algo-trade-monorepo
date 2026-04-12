-- Create the airflow metadata database if it doesn't exist.
-- This script runs on Postgres startup via docker-entrypoint-initdb.d.
SELECT 'CREATE DATABASE airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
