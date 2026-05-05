import React, { useMemo } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks';
import { 
  MetricCard, 
  StatusBadge, 
  DataTable, 
  Column 
} from '../components';
import { Stock } from '../types';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import styles from './Overview.module.css';

export const Overview: React.FC = () => {
  const { data: stocks, loading: stocksLoading } = useApi(() => api.listStocks());
  const { data: health, loading: healthLoading } = useApi(() => api.healthCheck());

  const sectorData = useMemo(() => {
    if (!stocks) return [];
    const counts: Record<string, number> = {};
    stocks.forEach(s => {
      if (s.sector && s.sector !== 'N/A') {
        counts[s.sector] = (counts[s.sector] || 0) + 1;
      }
    });
    return Object.entries(counts)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [stocks]);

  const sectorsCount = useMemo(() => {
    if (!stocks) return 0;
    return new Set(stocks.map(s => s.sector).filter(s => s && s !== 'N/A')).size;
  }, [stocks]);

  const stockColumns: Column<Stock>[] = [
    { header: 'Symbol', accessor: 'symbol', width: '100px' },
    { header: 'Company', accessor: 'name' },
    { header: 'Sector', accessor: 'sector' },
    { header: 'Exchange', accessor: 'exchange', width: '120px' },
  ];

  if (stocksLoading || healthLoading) return <div>Loading Overview...</div>;

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Overview</h1>
      </header>

      <div className={styles.metricsGrid}>
        <MetricCard 
          label="Tracked Stocks" 
          value={stocks?.length || 0} 
        />
        <MetricCard 
          label="Sectors" 
          value={sectorsCount} 
        />
        <MetricCard 
          label="API Status" 
          value={<StatusBadge status={health ? 'online' : 'offline'} />} 
        />
      </div>

      <div className={styles.layoutGrid}>
        <div className={styles.tableSection}>
          <h2>Coverage</h2>
          <DataTable 
            data={stocks || []} 
            columns={stockColumns} 
            height={500}
          />
        </div>

        <div className={styles.chartSection}>
          <h2>Sector Breakdown</h2>
          <div className={styles.chartContainer}>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={sectorData} layout="vertical" margin={{ left: 40, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="name" 
                  type="category" 
                  width={150} 
                  tick={{ fontSize: 12 }}
                />
                <Tooltip />
                <Bar dataKey="value" fill="#0284c7" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
