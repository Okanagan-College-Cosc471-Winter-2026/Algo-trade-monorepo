import React, { useState } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks';
import { DataTable, Column } from '../components';
import { Snapshot } from '../types';
import styles from './Snapshots.module.css';

export const Snapshots: React.FC = () => {
  const { data, loading, refresh } = useApi(() => api.listSnapshots());
  const [ticker, setTicker] = useState('ALL');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [format, setFormat] = useState<'parquet' | 'csv' | 'both'>('parquet');
  const [building, setBuilding] = useState(false);
  const [buildResult, setBuildResult] = useState<any>(null);

  const handleBuild = async (e: React.FormEvent) => {
    e.preventDefault();
    setBuilding(true);
    try {
      const result = await api.buildSnapshot({
        ticker,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        format,
      });
      setBuildResult(result);
      refresh();
    } catch (err) {
      console.error('Failed to build snapshot', err);
    } finally {
      setBuilding(false);
    }
  };

  const handleDownload = async (filename: string) => {
    try {
      const blob = await api.downloadSnapshot(filename);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Failed to download snapshot', err);
    }
  };

  const columns: Column<Snapshot>[] = [
    { header: 'Filename', accessor: 'filename' },
    { 
      header: 'Size (MB)', 
      accessor: (s) => s.size_mb.toFixed(2), 
      width: '120px', 
      align: 'right' 
    },
    {
      header: 'Actions',
      accessor: (s) => (
        <button 
          className={styles.downloadBtn}
          onClick={() => handleDownload(s.filename)}
        >
          Download
        </button>
      ),
      width: '120px',
      align: 'center'
    }
  ];

  if (loading && !data) return <div>Loading Snapshots...</div>;

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Dataset Snapshots</h1>
      </header>

      <section className={styles.section}>
        <h2>Build New Snapshot</h2>
        <form className={styles.form} onSubmit={handleBuild}>
          <div className={styles.formGroup}>
            <label>Ticker</label>
            <input 
              type="text" 
              value={ticker} 
              onChange={(e) => setTicker(e.target.value)} 
              placeholder="ALL or specific symbol"
            />
          </div>
          <div className={styles.formRow}>
            <div className={styles.formGroup}>
              <label>Start Date</label>
              <input 
                type="date" 
                value={startDate} 
                onChange={(e) => setStartDate(e.target.value)} 
              />
            </div>
            <div className={styles.formGroup}>
              <label>End Date</label>
              <input 
                type="date" 
                value={endDate} 
                onChange={(e) => setEndDate(e.target.value)} 
              />
            </div>
          </div>
          <div className={styles.formGroup}>
            <label>Format</label>
            <select value={format} onChange={(e) => setFormat(e.target.value as any)}>
              <option value="parquet">Parquet</option>
              <option value="csv">CSV</option>
              <option value="both">Both</option>
            </select>
          </div>
          <button type="submit" className={styles.buildBtn} disabled={building}>
            {building ? 'Building...' : 'Build Snapshot'}
          </button>
        </form>
        {buildResult && (
          <div className={styles.result}>
            <h3>Build Success</h3>
            <pre>{JSON.stringify(buildResult, null, 2)}</pre>
          </div>
        )}
      </section>

      <section className={styles.section}>
        <h2>Available Snapshots</h2>
        <p className={styles.directory}>Directory: {data?.directory || '—'}</p>
        <DataTable 
          data={data?.snapshots || []} 
          columns={columns} 
          height={400}
        />
      </section>
    </div>
  );
};
