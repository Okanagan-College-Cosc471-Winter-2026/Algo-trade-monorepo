import React, { useState } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks';
import { 
  CandlestickChart, 
  MetricCard, 
  StatusBadge 
} from '../components';
import { Prediction } from '../types';
import styles from './Predictions.module.css';

export const Predictions: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);

  const { data: stocks } = useApi(() => api.listStocks());
  const { data: ohlc } = useApi(() => api.getOHLC(symbol, 90), [symbol]);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const res = await api.predict(symbol);
      setPrediction(res);
    } catch (err) {
      console.error('Prediction failed', err);
    } finally {
      setLoading(false);
    }
  };

  const formattedPred = React.useMemo(() => {
    if (!prediction || !prediction.path) return undefined;
    const date = prediction.prediction_date.split('T')[0];
    return prediction.path.map(p => ({
      time: Math.floor(new Date(`${date}T${p.bar_time}:00Z`).getTime() / 1000),
      value: p.pred_close
    }));
  }, [prediction]);

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Inference & Forecasting</h1>
      </header>

      <section className={styles.controlSection}>
        <div className={styles.controls}>
          <select 
            value={symbol} 
            onChange={(e) => {
              setSymbol(e.target.value);
              setPrediction(null);
            }}
            className={styles.symbolSelect}
          >
            {stocks?.map(s => (
              <option key={s.symbol} value={s.symbol}>
                {s.symbol} - {s.name}
              </option>
            ))}
          </select>
          <button 
            onClick={handleGenerate} 
            disabled={loading}
            className={styles.generateBtn}
          >
            {loading ? 'Running Inference...' : 'Generate Prediction'}
          </button>
        </div>
      </section>

      {prediction && (
        <div className={styles.resultSection}>
          <div className={styles.metricsGrid}>
            <MetricCard 
              label="Latest Price" 
              value={`$${prediction.current_price.toFixed(2)}`} 
            />
            <MetricCard 
              label="Predicted EOD" 
              value={`$${(prediction.path[prediction.path.length - 1]?.pred_close || 0).toFixed(2)}`} 
            />
            <MetricCard 
              label="Expected Return" 
              value={`${prediction.predicted_full_day_return.toFixed(2)}%`}
              delta={prediction.predicted_direction.toUpperCase()}
              deltaType={prediction.predicted_direction === 'up' ? 'increase' : 'decrease'}
            />
            <MetricCard 
              label="Model Version" 
              value={prediction.model_version.split('_').slice(0, 2).join(' ')} 
            />
          </div>

          <div className={styles.chartWrapper}>
            <h3>{symbol} — Predicted Path for {prediction.prediction_date.split('T')[0]}</h3>
            <CandlestickChart 
              data={ohlc || []} 
              warmPrediction={formattedPred}
            />
          </div>
        </div>
      )}

      {!prediction && !loading && (
        <div className={styles.emptyState}>
          <div className={styles.icon}>🎯</div>
          <h2>Ready for Inference</h2>
          <p>Select a stock and click "Generate Prediction" to run the XG-Boost model on the latest features.</p>
        </div>
      )}
    </div>
  );
};
