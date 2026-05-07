import React, { useState, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import { useApi, useInterval } from '../hooks';
import {
  StatusBadge,
  ProgressBar,
  LogViewer,
  DataTable,
  MetricCard,
  Column
} from '../components';
import { AirflowStatus } from '../types';
import styles from './Ops.module.css';

function stateColor(state: string | null): string {
  if (!state) return '#94a3b8';
  const s = state.toLowerCase();
  if (s === 'success') return '#16a34a';
  if (s === 'running') return '#0284c7';
  if (s === 'failed') return '#dc2626';
  if (s === 'queued') return '#d97706';
  return '#64748b';
}

function fmtUTC(ts: string | null): string {
  if (!ts) return '—';
  return ts.slice(0, 16).replace('T', ' ');
}

function fmtDuration(s: number | null): string {
  if (s == null) return '—';
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return rem ? `${m}m ${rem}s` : `${m}m`;
}

function cleanSchedule(raw: string | null): string {
  if (!raw || raw === 'null') return 'manual';
  return raw.replace(/^"|"$/g, '');
}

export const Ops: React.FC = () => {
  const { data: status, loading, refresh: refreshStatus } = useApi(() => api.opsStatus());
  const { data: airflow, refresh: refreshAirflow } = useApi(() => api.opsAirflow());
  const [logs, setLogs] = useState<string[]>([]);
  const [logType, setLogType] = useState<'pipeline_15m' | 'nibi_usage'>('pipeline_15m');
  const [nibiCmd, setNibiCmd] = useState('');
  const [nibiResult, setNibiResult] = useState<{ rc: number; stdout: string; stderr: string } | null>(null);
  const [showPausedDags, setShowPausedDags] = useState(false);

  // Poll status every 30s, airflow every 60s
  useInterval(refreshStatus, 30_000);
  useInterval(refreshAirflow, 60_000);

  // Fetch logs separately
  const fetchLogs = useCallback(async () => {
    try {
      const resp = await api.opsLogTail(logType, 60);
      if (resp.lines) setLogs(resp.lines);
    } catch (err) {
      console.error('Failed to fetch logs', err);
    }
  }, [logType]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useInterval(fetchLogs, 10000);

  if (loading && !status) return <div>Loading Ops Status...</div>;
  if (!status) return <div>Error loading status.</div>;

  const handleRunNibiCmd = async () => {
    if (!nibiCmd.trim()) return;
    try {
      const res = await api.opsNibiExec(nibiCmd);
      setNibiResult(res);
    } catch (err) {
      console.error('Failed to run NIBI command', err);
    }
  };

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>System Operations</h1>
        <span className={styles.timestamp}>
          Snapshot at {status.generated_at.replace('T', ' ').slice(0, 19)} UTC
        </span>
      </header>

      {!status.ssh_socket.alive && (
        <div className={styles.alert}>
          <strong>NIBI SSH socket is DOWN</strong> — model training is unavailable.
          Run <code>morning_login.sh</code> on the host machine.
        </div>
      )}

      <section className={styles.section}>
        <h2>Service Health</h2>
        <div className={styles.grid}>
          <MetricCard 
            label="Backend API" 
            value={<StatusBadge status="online" />} 
          />
          <MetricCard 
            label="NIBI SSH Socket" 
            value={<StatusBadge status={status.ssh_socket.alive ? 'online' : 'offline'} label={status.ssh_socket.alive ? 'Socket Alive' : 'Socket Dead'} />} 
          />
          <MetricCard 
            label="Collector Pipeline" 
            value={<StatusBadge status={status.collector.collector_state} />} 
            help={status.collector.error}
          />
          <MetricCard 
            label="Market Data" 
            value={<StatusBadge status={status.data.freshness_state} />} 
            help={status.data.freshness_reason}
          />
        </div>
      </section>

      {/* ── Airflow Pipeline Status ── */}
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2>Pipeline Schedule</h2>
          <button className={styles.toggleSmall} onClick={() => setShowPausedDags(v => !v)}>
            {showPausedDags ? 'Hide paused' : 'Show paused'}
          </button>
        </div>

        {airflow?.error && (
          <div className={styles.alertWarn}>Airflow API: {airflow.error}</div>
        )}

        <div className={styles.dagTable}>
          <div className={styles.dagHeader}>
            <span>DAG</span>
            <span>Schedule</span>
            <span>Last State</span>
            <span>Last Run (UTC)</span>
            <span>Duration</span>
            <span>Next Run (UTC)</span>
          </div>
          {(airflow?.dags ?? [])
            .filter(d => showPausedDags ? true : !d.is_paused)
            .map(dag => {
              const dur = dag.last_start && dag.last_end
                ? Math.round((new Date(dag.last_end).getTime() - new Date(dag.last_start).getTime()) / 1000)
                : null;
              return (
                <div
                  key={dag.dag_id}
                  className={`${styles.dagRow} ${dag.is_paused ? styles.dagPaused : ''}`}
                >
                  <span className={styles.dagId}>
                    {dag.dag_id.replace(/_/g, '_​')}
                    {dag.is_paused && <span className={styles.pausedBadge}>paused</span>}
                  </span>
                  <span className={styles.dagSchedule}>{cleanSchedule(dag.schedule)}</span>
                  <span style={{ color: stateColor(dag.last_state), fontWeight: 600 }}>
                    {dag.last_state ?? '—'}
                  </span>
                  <span>{fmtUTC(dag.last_start)}</span>
                  <span>{fmtDuration(dur)}</span>
                  <span className={dag.next_run ? '' : styles.metaMuted}>{fmtUTC(dag.next_run)}</span>
                </div>
              );
            })
          }
        </div>

        {/* Recent run history */}
        {(airflow?.recent_runs?.length ?? 0) > 0 && (
          <div style={{ marginTop: '1.25rem' }}>
            <h3 className={styles.subHeading}>Recent Runs</h3>
            <div className={styles.runTable}>
              <div className={styles.runHeader}>
                <span>DAG</span>
                <span>Type</span>
                <span>State</span>
                <span>Started (UTC)</span>
                <span>Duration</span>
              </div>
              {(airflow?.recent_runs ?? []).slice(0, 15).map((r, i) => (
                <div key={i} className={styles.runRow}>
                  <span className={styles.dagId}>{r.dag_id.replace(/_/g, '_​')}</span>
                  <span className={styles.runType}>{r.run_type}</span>
                  <span style={{ color: stateColor(r.state), fontWeight: 600 }}>{r.state}</span>
                  <span>{fmtUTC(r.started)}</span>
                  <span>{fmtDuration(r.duration_s)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      <section className={styles.section}>
        <h2>NIBI Training Job</h2>
        <div className={styles.jobInfo}>
          <div className={styles.grid}>
            <MetricCard label="Job ID" value={status.live_job_primary?.job_id || status.nibi_job.job_id || '—'} />
            <MetricCard label="Job Name" value={status.live_job_primary?.name || '—'} />
            <MetricCard label="Status" value={<StatusBadge status={status.live_job_primary?.state || status.nibi_job.live_state || 'unknown'} />} />
            <MetricCard label="Elapsed/Limit" value={status.live_job_primary ? `${status.live_job_primary.elapsed} / ${status.live_job_primary.time_lim}` : '—'} />
          </div>
          
          <div className={styles.progressSection}>
            <ProgressBar 
              value={status.model.windows_ok / (status.model.windows_total || 1)} 
              label={`Warm-refresh windows: ${status.model.windows_ok} / ${status.model.windows_total} completed`}
              subLabel={status.model.windows_error ? `${status.model.windows_error} errors` : ''}
              variant="primary"
            />
          </div>
        </div>
      </section>

      <section className={styles.section}>
        <h2>Live Logs</h2>
        <div className={styles.logControls}>
          <button 
            className={logType === 'pipeline_15m' ? styles.activeTab : ''} 
            onClick={() => setLogType('pipeline_15m')}
          >
            Data Pipeline
          </button>
          <button 
            className={logType === 'nibi_usage' ? styles.activeTab : ''} 
            onClick={() => setLogType('nibi_usage')}
          >
            NIBI Usage
          </button>
        </div>
        <LogViewer lines={logs} />
      </section>

      <section className={styles.section}>
        <h2>NIBI Remote Commands</h2>
        <div className={styles.nibiConsole}>
          <input 
            type="text" 
            value={nibiCmd} 
            onChange={(e) => setNibiCmd(e.target.value)}
            placeholder="squeue -u harshsaw"
            className={styles.cmdInput}
          />
          <button 
            onClick={handleRunNibiCmd}
            disabled={!status.ssh_socket.alive}
            className={styles.runBtn}
          >
            Run on NIBI
          </button>
        </div>
        {nibiResult && (
          <div className={styles.consoleOutput}>
            <div className={styles.rc}>Exit code: {nibiResult.rc}</div>
            <pre>{nibiResult.stdout}</pre>
            {nibiResult.stderr && <pre className={styles.stderr}>{nibiResult.stderr}</pre>}
          </div>
        )}
      </section>
    </div>
  );
};
