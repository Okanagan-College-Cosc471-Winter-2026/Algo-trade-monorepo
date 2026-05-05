import React, { useState, useMemo } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks';
import { 
  CandlestickChart, 
  MetricCard, 
  ProgressBar 
} from '../components';
import { SimStepPrediction } from '../types';
import styles from './Simulation.module.css';

export const Simulation: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [currentStep, setCurrentStep] = useState(0);
  const [mode, setMode] = useState<'base' | 'warm'>('base');

  const { data: symbols } = useApi(() => api.simSymbols());
  const { data: session } = useApi(() => api.simSession());
  const { data: history } = useApi(() => api.simHistory(selectedSymbol), [selectedSymbol]);
  const { data: ohlc } = useApi(() => api.simOHLC(selectedSymbol), [selectedSymbol]);
  
  const { data: basePred } = useApi(
    () => api.simBase(selectedSymbol), 
    [selectedSymbol]
  );
  
  const { data: stepPred } = useApi(
    () => api.simStep(selectedSymbol, currentStep), 
    [selectedSymbol, currentStep]
  );

  const activePred = mode === 'warm' ? stepPred : basePred;

  const formattedHistory = useMemo(() => {
    return history?.map(d => ({
      time: d.time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    })) || [];
  }, [history]);

  const formattedPrediction = useMemo(() => {
    if (!activePred || !ohlc || ohlc.length === 0) return undefined;
    
    // In simulation, predictions are often log returns relative to an anchor
    // For simplicity here, we'll map them to the same timestamps as the simulation day OHLC
    const anchorPrice = mode === 'warm' && currentStep < ohlc.length 
      ? ohlc[currentStep].close 
      : ohlc[0].open;

    return activePred.bars.map((b, i) => {
      const ohlcIdx = mode === 'warm' ? currentStep + i : i;
      if (ohlcIdx >= ohlc.length) return null;
      return {
        time: ohlc[ohlcIdx].time,
        value: anchorPrice * Math.exp(b.pred_log_return)
      };
    }).filter(Boolean) as { time: number; value: number }[];
  }, [activePred, ohlc, mode, currentStep]);

  if (!session) return <div>Loading Simulation...</div>;

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.titleGroup}>
          <h1>Simulation</h1>
          <select 
            value={selectedSymbol} 
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className={styles.symbolSelect}
          >
            {symbols?.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
        <div className={styles.modeToggle}>
          <button 
            className={mode === 'base' ? styles.activeMode : ''} 
            onClick={() => setMode('base')}
          >
            Base Model
          </button>
          <button 
            className={mode === 'warm' ? styles.activeMode : ''} 
            onClick={() => setMode('warm')}
          >
            Warm Refresh
          </button>
        </div>
      </header>

      <div className={styles.metricsGrid}>
        <MetricCard label="Replay Date" value={session.replay_date} />
        <MetricCard label="Base Trained" value={session.effective_as_of_date} />
        <MetricCard 
          label="Predicted Return" 
          value={`${activePred?.predicted_full_day_return.toFixed(4)}%`} 
        />
        <MetricCard 
          label="Direction" 
          value={activePred?.predicted_direction.toUpperCase() || '—'} 
        />
      </div>

      {mode === 'warm' && (
        <div className={styles.sliderSection}>
          <div className={styles.sliderHeader}>
            <span>Step: {currentStep} ({session.step_labels[currentStep]})</span>
            <span>Total Trees: {session.base_trees + (currentStep + 1) * session.warm_trees_per_step}</span>
          </div>
          <input 
            type="range" 
            min={0} 
            max={session.steps_completed - 1} 
            value={currentStep} 
            onChange={(e) => setCurrentStep(parseInt(e.target.value))}
            className={styles.slider}
          />
        </div>
      )}

      <div className={styles.chartWrapper}>
        <CandlestickChart 
          data={formattedHistory.concat(ohlc || [])} 
          warmPrediction={mode === 'warm' ? formattedPrediction : undefined}
          basePrediction={mode === 'base' ? formattedPrediction : undefined}
        />
      </div>
    </div>
  );
};
