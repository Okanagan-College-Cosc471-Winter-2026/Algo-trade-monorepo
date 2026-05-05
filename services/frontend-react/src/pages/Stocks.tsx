import React, { useState, useEffect, useMemo } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks';
import { 
  CandlestickChart, 
  MetricCard, 
  DataTable, 
  Column 
} from '../components';
import { OHLC, Stock, Prediction } from '../types';
import styles from './Stocks.module.css';

export const Stocks: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [days, setDays] = useState(30);
  const [showPredictions, setShowPredictions] = useState(false);

  const { data: stocks } = useApi(() => api.listStocks());
  const { data: ohlc, loading: ohlcLoading } = useApi(
    () => api.getOHLC(selectedSymbol, days), 
    [selectedSymbol, days]
  );

  const [basePred, setBasePred] = useState<Prediction | null>(null);
  const [warmPred, setWarmPred] = useState<Prediction | null>(null);

  const selectedStock = useMemo(() => 
    stocks?.find(s => s.symbol === selectedSymbol), 
    [stocks, selectedSymbol]
  );

  useEffect(() => {
    if (showPredictions) {
      api.predictBase(selectedSymbol).then(setBasePred).catch(console.error);
      api.predict(selectedSymbol).then(setWarmPred).catch(console.error);
    } else {
      setBasePred(null);
      setWarmPred(null);
    }
  }, [showPredictions, selectedSymbol]);

  const formattedBasePred = useMemo(() => {
    if (!basePred || !basePred.path) return undefined;
    const date = basePred.prediction_date.split('T')[0];
    return basePred.path.map(p => ({
      time: Math.floor(new Date(`${date}T${p.bar_time}:00Z`).getTime() / 1000),
      value: p.pred_close
    }));
  }, [basePred]);

  const formattedWarmPred = useMemo(() => {
    if (!warmPred || !warmPred.path) return undefined;
    const date = warmPred.prediction_date.split('T')[0];
    return warmPred.path.map(p => ({
      time: Math.floor(new Date(`${date}T${p.bar_time}:00Z`).getTime() / 1000),
      value: p.pred_close
    }));
  }, [warmPred]);

  const stats = useMemo(() => {
    if (!ohlc || ohlc.length === 0) return null;
    const latest = ohlc[ohlc.length - 1];
    const prev = ohlc[ohlc.length - 2] || latest;
    const change = latest.close - prev.close;
    const changePct = (change / prev.close) * 100;
    const high = Math.max(...ohlc.map(d => d.high));
    const low = Math.min(...ohlc.map(d => d.low));
    
    return {
      latest: latest.close,
      change,
      changePct,
      high,
      low,
      volume: latest.volume
    };
  }, [ohlc]);

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.titleGroup}>
          <h1>Stocks</h1>
          <select 
            value={selectedSymbol} 
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className={styles.symbolSelect}
          >
            {stocks?.map(s => (
              <option key={s.symbol} value={s.symbol}>
                {s.symbol} - {s.name}
              </option>
            ))}
          </select>
        </div>
        <div className={styles.controls}>
          <div className={styles.rangeButtons}>
            {[7, 30, 90, 180, 365].map(d => (
              <button 
                key={d}
                onClick={() => setDays(d)}
                className={days === d ? styles.activeRange : ''}
              >
                {d}D
              </button>
            ))}
          </div>
          <label className={styles.toggle}>
            <input 
              type="checkbox" 
              checked={showPredictions} 
              onChange={(e) => setShowPredictions(e.target.checked)}
            />
            Show Predictions
          </label>
        </div>
      </header>

      {stats && (
        <div className={styles.metricsGrid}>
          <MetricCard 
            label="Last Close" 
            value={`$${stats.latest.toFixed(2)}`}
            delta={`${stats.change > 0 ? '+' : ''}${stats.change.toFixed(2)} (${stats.changePct.toFixed(2)}%)`}
            deltaType={stats.change >= 0 ? 'increase' : 'decrease'}
          />
          <MetricCard 
            label={`${days}D High`} 
            value={`$${stats.high.toFixed(2)}`}
          />
          <MetricCard 
            label={`${days}D Low`} 
            value={`$${stats.low.toFixed(2)}`}
          />
          <MetricCard 
            label="Latest Volume" 
            value={stats.volume.toLocaleString()}
          />
        </div>
      )}

      <div className={styles.chartWrapper}>
        {ohlcLoading ? (
          <div className={styles.loading}>Loading Chart...</div>
        ) : (
          <CandlestickChart 
            data={ohlc || []} 
            basePrediction={formattedBasePred}
            warmPrediction={formattedWarmPred}
          />
        )}
      </div>

      <div className={styles.infoSection}>
        <div className={styles.card}>
          <h2>Company Information</h2>
          <div className={styles.infoGrid}>
            <div><strong>Sector:</strong> {selectedStock?.sector || 'N/A'}</div>
            <div><strong>Industry:</strong> {selectedStock?.industry || 'N/A'}</div>
            <div><strong>Exchange:</strong> {selectedStock?.exchange || 'N/A'}</div>
          </div>
        </div>
        
        {showPredictions && (basePred || warmPred) && (
          <div className={styles.card}>
            <h2>Prediction Metadata</h2>
            <div className={styles.infoGrid}>
              {basePred && (
                <div>
                  <strong>Base Model:</strong> {basePred.model_version}<br/>
                  <strong>Target:</strong> {basePred.prediction_date.split('T')[0]}
                </div>
              )}
              {warmPred && (
                <div>
                  <strong>Warm Refresh:</strong> {warmPred.refreshed_at?.slice(11, 16) || 'N/A'} UTC<br/>
                  <strong>Refreshed As Of:</strong> {warmPred.as_of?.slice(11, 16) || 'N/A'} UTC
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
