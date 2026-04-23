Restore checklist:
1) Ensure docker compose stack is up (db service running).
2) Restore DB:
   docker compose exec -T db dropdb -U appuser --if-exists algotrade
   docker compose exec -T db createdb -U appuser algotrade
   cat db/algotrade.dump | docker compose exec -T db pg_restore -U appuser -d algotrade --clean --if-exists
3) Restore model artifacts:
   tar -xzf artifacts/model_artifacts.tar.gz -C /data/projects/Algo-trade-monorepo
4) Verify symlinks:
   ls -la /data/projects/Algo-trade-monorepo/model_artifacts/current_base
5) Restart services:
   docker compose up -d backend frontend scheduler collector
