import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks';
import { CandlestickChart, MetricCard } from '../components';
import styles from './Simulation.module.css';

// Returns current 15-min step index (0–25) based on ET wall clock.
// Step 0 = 09:30 ET bar, step 25 = 15:45 ET bar.
function computeLiveStep(): number {
  const now = new Date();
  const etStr = now.toLocaleString('en-US', { timeZone: 'America/New_York', hour12: false,
    hour: '2-digit', minute: '2-digit' });
  const [h, m] = etStr.split(':').map(Number);
  const totalMins = h * 60 + m;
  const openMins = 9 * 60 + 30;
  const closeMins = 15 * 60 + 45;
  if (totalMins < openMins) return 0;
  if (totalMins >= closeMins) return 25;
  return Math.min(25, Math.floor((totalMins - openMins) / 15));
}

// Convert step index to market time: 9:30 AM + (step * 15 minutes)
function stepToMarketTime(step: number): string {
  const baseHour = 9;
  const baseMin = 30;
  const totalMins = baseHour * 60 + baseMin + step * 15;
  const h = Math.floor(totalMins / 60);
  const m = totalMins % 60;
  return `${h}:${m.toString().padStart(2, '0')}`;
}

// Returns today's date as YYYY-MM-DD if it looks like a trading day (Mon–Fri),
// otherwise returns null.
function todayIfTradingDay(): string | null {
  const now = new Date();
  const dow = now.toLocaleDateString('en-US', { timeZone: 'America/New_York', weekday: 'short' });
  if (dow === 'Sat' || dow === 'Sun') return null;
  return now.toLocaleDateString('en-CA', { timeZone: 'America/New_York' }); // YYYY-MM-DD
}

export const Simulation: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [currentStep, setCurrentStep] = useState(computeLiveStep);
  const [manualStep, setManualStep] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  const liveDate = todayIfTradingDay();

  const {
    data: symbols,
    error: symbolsError,
    loading: symbolsLoading,
  } = useApi(() => api.simSymbols(), [refreshKey]);
  const { data: session } = useApi(() => api.simSession(), [refreshKey]);

  // History: last 5 trading days up to replay_date (context candles)
  const { data: history } = useApi(() => api.simHistory(selectedSymbol), [selectedSymbol, refreshKey]);

  // OHLC: today's live bars if it's a trading day, else replay_date bars
  const { data: ohlc } = useApi(
    () => api.simOHLC(selectedSymbol, liveDate ?? undefined),
    [selectedSymbol, liveDate, refreshKey]
  );

  // Base prediction (always shown – amber dotted line)
  const { data: basePred } = useApi(
    () => api.simBase(selectedSymbol),
    [selectedSymbol, refreshKey]
  );

  // Warm prediction for current step (always shown – blue dashed line)
  const { data: stepPred } = useApi(
    () => api.simStep(selectedSymbol, currentStep),
    [selectedSymbol, currentStep, refreshKey]
  );

  // Auto-advance step to wall clock every minute; auto-refresh data every 15 min
  useEffect(() => {
    const stepTimer = setInterval(() => {
      if (!manualStep) setCurrentStep(computeLiveStep());
    }, 60_000);

    const refreshTimer = setInterval(() => {
      setRefreshKey(k => k + 1);
      if (!manualStep) setCurrentStep(computeLiveStep());
    }, 15 * 60_000);

    return () => {
      clearInterval(stepTimer);
      clearInterval(refreshTimer);
    };
  }, [manualStep]);

  // Keyboard navigation for manual step replay
  const maxStepIndex = session ? Math.max(0, session.steps_completed - 1) : 25;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setManualStep(true);
        setCurrentStep(s => Math.max(0, s - 1));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setManualStep(true);
        setCurrentStep(s => Math.min(maxStepIndex, s + 1));
      } else if (e.key === 'l' || e.key === 'L') {
        setManualStep(false);
        setCurrentStep(computeLiveStep());
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [maxStepIndex]);

  useEffect(() => {
    if (!symbols?.length) return;
    if (!symbols.includes(selectedSymbol)) setSelectedSymbol(symbols[0]);
  }, [symbols, selectedSymbol]);

  // Context candles: history (prior days) + today's live bars up to currentStep (progressive)
  const chartCandles = useMemo(() => {
    const histBars = (history ?? []).map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close }));
    // Only show today's bars up to currentStep (progressive reveal)
    const todayBars = (ohlc ?? []).slice(0, currentStep + 1).map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close }));
    const seen = new Set(histBars.map(b => b.time));
    const merged = [...histBars, ...todayBars.filter(b => !seen.has(b.time))];
    return merged.sort((a, b) => a.time - b.time);
  }, [history, ohlc, currentStep]);

  // Base prediction line — anchored to first OHLC open
  const formattedBasePred = useMemo(() => {
    if (!basePred || !ohlc || ohlc.length === 0) return undefined;
    const anchor = ohlc[0].open;
    return basePred.bars
      .map((b, i) => ohlc[i] ? { time: ohlc[i].time, value: anchor * Math.exp(b.pred_log_return) } : null)
      .filter(Boolean) as { time: number; value: number }[];
  }, [basePred, ohlc]);

  // Warm prediction line — anchored to current step's close
  const formattedWarmPred = useMemo(() => {
    if (!stepPred || !ohlc || ohlc.length === 0) return undefined;
    const anchorIdx = Math.min(currentStep, ohlc.length - 1);
    const anchor = ohlc[anchorIdx]?.close ?? ohlc[0].open;
    return stepPred.bars
      .map((b, i) => {
        const ohlcIdx = currentStep + i;
        return ohlc[ohlcIdx] ? { time: ohlc[ohlcIdx].time, value: anchor * Math.exp(b.pred_log_return) } : null;
      })
      .filter(Boolean) as { time: number; value: number }[];
  }, [stepPred, ohlc, currentStep]);

  // Compute zoom bounds: show 5-day window with ~40% of width as prediction zone
  const visibleFrom = useMemo(() => {
    if (!chartCandles.length) return undefined;
    // 5 trading days back = ~35 calendar days, but we use history length as proxy
    return chartCandles[Math.max(0, chartCandles.length - 33)]?.time;
  }, [chartCandles]);

  // futureStart = first timestamp where prediction takes over (step after currentStep)
  const futureStart = useMemo(() => {
    if (!ohlc || ohlc.length <= currentStep) return undefined;
    return ohlc[currentStep]?.time;
  }, [ohlc, currentStep]);

  if (!session) return <div>Loading Simulation…</div>;

  const symbolsReady = Array.isArray(symbols) && symbols.length > 0;
  const stepLabel = session.step_labels?.[currentStep] ?? `Step ${currentStep}`;
  const isLive = !manualStep && liveDate !== null;

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.titleGroup}>
          <h1>
            Simulation
            {isLive && <span className={styles.liveTag}>● LIVE</span>}
          </h1>
          {symbolsLoading && <span className={styles.symbolStatus}>Loading symbols…</span>}
          {!symbolsLoading && symbolsError && (
            <span className={styles.symbolError} title={symbolsError.message}>
              Failed to load symbols: {symbolsError.message}
            </span>
          )}
          {!symbolsLoading && !symbolsError && Array.isArray(symbols) && symbols.length === 0 && (
            <span className={styles.symbolError}>
              No symbols in artifact bundle — check backend{' '}
              <code>model_artifacts/current_simulation</code>
            </span>
          )}
          {symbolsReady && (
            <select
              value={symbols.includes(selectedSymbol) ? selectedSymbol : symbols[0]}
              onChange={e => setSelectedSymbol(e.target.value)}
              className={styles.symbolSelect}
              aria-label="Ticker symbol"
            >
              {symbols.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          )}
        </div>
      </header>

      <div className={styles.metricsGrid}>
        <MetricCard label="Replay Date" value={session.replay_date} />
        <MetricCard label="Base Trained" value={session.effective_as_of_date} />
        <MetricCard
          label="Base Return"
          value={basePred?.predicted_full_day_return != null
            ? `${(basePred.predicted_full_day_return * 100).toFixed(2)}%` : '—'}
        />
        <MetricCard
          label={`Warm Return (${stepLabel})`}
          value={stepPred?.predicted_full_day_return != null
            ? `${(stepPred.predicted_full_day_return * 100).toFixed(2)}%` : '—'}
        />
        <MetricCard
          label="Direction"
          value={stepPred?.predicted_direction?.toUpperCase() ?? basePred?.predicted_direction?.toUpperCase() ?? '—'}
        />
      </div>

      <div className={styles.sliderSection}>
        <div className={styles.sliderHeader}>
          <span className={styles.timeDisplay}>
            Market Time: <strong>{stepToMarketTime(currentStep)}</strong>
            {manualStep && (
              <button className={styles.liveBtn} onClick={() => { setManualStep(false); setCurrentStep(computeLiveStep()); }}>
                ↺ Live
              </button>
            )}
          </span>
          <span className={styles.legend}>
            <span className={styles.legendBase}>── Base forecast (static)</span>
            <span className={styles.legendWarm}>─ ─ Warm refresh (updates every 15 min)</span>
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={maxStepIndex}
          value={Math.min(currentStep, maxStepIndex)}
          onChange={e => { setManualStep(true); setCurrentStep(parseInt(e.target.value, 10)); }}
          className={styles.slider}
          title={stepToMarketTime(Math.min(currentStep, maxStepIndex))}
        />
        <div className={styles.timeLabels}>
          <span className={styles.timeLabel}>Market Open<br/>9:30</span>
          <span className={styles.timeLabel}>Mid-Day<br/>12:00</span>
          <span className={styles.timeLabel}>Market Close<br/>15:45</span>
        </div>
        <div className={styles.keyboardHint}>
          <kbd>←</kbd> <kbd>→</kbd> navigate &nbsp;|&nbsp; <kbd>L</kbd> return to live
        </div>
      </div>

      <div className={styles.chartWrapper}>
        <CandlestickChart
          data={chartCandles}
          basePrediction={formattedBasePred}
          warmPrediction={formattedWarmPred}
          visibleFrom={visibleFrom}
          futureStart={futureStart}
          currentStep={currentStep}
        />
      </div>
    </div>
  );
};
