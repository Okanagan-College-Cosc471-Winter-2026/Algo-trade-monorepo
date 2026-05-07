import React, { useState, useMemo, useEffect } from 'react';
import { api } from '../api/client';
import { useApi } from '../hooks';
import { CandlestickChart, MetricCard } from '../components';
import styles from './Simulation.module.css';

// Step 0 = 09:30 ET, step 25 = 15:45 ET (15-min bars)
function computeLiveStep(): number {
  const now = new Date();
  const etStr = now.toLocaleString('en-US', {
    timeZone: 'America/New_York', hour12: false,
    hour: '2-digit', minute: '2-digit',
  });
  const [h, m] = etStr.split(':').map(Number);
  const totalMins = h * 60 + m;
  const openMins = 9 * 60 + 30;
  const closeMins = 15 * 60 + 45;
  if (totalMins < openMins) return 0;
  if (totalMins >= closeMins) return 25;
  return Math.min(25, Math.floor((totalMins - openMins) / 15));
}

function stepToMarketTime(step: number): string {
  const totalMins = 9 * 60 + 30 + step * 15;
  const h = Math.floor(totalMins / 60);
  const m = totalMins % 60;
  return `${h}:${m.toString().padStart(2, '0')}`;
}

function todayIfTradingDay(): string | null {
  const now = new Date();
  const dow = now.toLocaleDateString('en-US', { timeZone: 'America/New_York', weekday: 'short' });
  if (dow === 'Sat' || dow === 'Sun') return null;
  return now.toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

export const Simulation: React.FC = () => {
  // MU (Micron): base called DOWN, warm refresh flipped to UP +3.33% — best warm-refresh demo
  const [selectedSymbol, setSelectedSymbol] = useState('MU');
  const [currentStep, setCurrentStep] = useState(computeLiveStep);
  const [manualStep, setManualStep] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);

  const liveDate = todayIfTradingDay();

  const { data: symbols, error: symbolsError, loading: symbolsLoading } =
    useApi(() => api.simSymbols(), [refreshKey]);
  const { data: session } = useApi(() => api.simSession(), [refreshKey]);
  const { data: history } = useApi(() => api.simHistory(selectedSymbol), [selectedSymbol, refreshKey]);
  const { data: ohlc } = useApi(
    () => api.simOHLC(selectedSymbol, liveDate ?? undefined),
    [selectedSymbol, liveDate, refreshKey]
  );
  const { data: basePred } = useApi(() => api.simBase(selectedSymbol), [selectedSymbol, refreshKey]);
  const { data: stepPred } = useApi(
    () => api.simStep(selectedSymbol, currentStep),
    [selectedSymbol, currentStep, refreshKey]
  );

  // Auto-advance step every minute; refresh data every 15 min
  useEffect(() => {
    const stepTimer = setInterval(() => {
      if (!manualStep) setCurrentStep(computeLiveStep());
    }, 60_000);
    const refreshTimer = setInterval(() => {
      setRefreshKey(k => k + 1);
      if (!manualStep) setCurrentStep(computeLiveStep());
    }, 15 * 60_000);
    return () => { clearInterval(stepTimer); clearInterval(refreshTimer); };
  }, [manualStep]);

  // Keyboard navigation
  const maxStepIndex = session ? Math.max(0, session.steps_completed - 1) : 25;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (e.key === 'ArrowUp' || e.key === 'ArrowRight') {
        e.preventDefault();
        setManualStep(true);
        setCurrentStep(s => Math.min(maxStepIndex, s + 1));
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') {
        e.preventDefault();
        setManualStep(true);
        setCurrentStep(s => Math.max(0, s - 1));
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

  // Progressive candles: history (prior days) + today bars up to currentStep
  const chartCandles = useMemo(() => {
    const ohlcTimes = new Set((ohlc ?? []).map(b => b.time));
    const histBars = (history ?? [])
      .filter(d => !ohlcTimes.has(d.time))
      .map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close }));
    const todayBars = (ohlc ?? []).slice(0, currentStep + 1)
      .map(d => ({ time: d.time, open: d.open, high: d.high, low: d.low, close: d.close }));
    return [...histBars, ...todayBars].sort((a, b) => a.time - b.time);
  }, [history, ohlc, currentStep]);

  // Base prediction line — anchored to replay day open
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
        const idx = currentStep + i;
        return ohlc[idx] ? { time: ohlc[idx].time, value: anchor * Math.exp(b.pred_log_return) } : null;
      })
      .filter(Boolean) as { time: number; value: number }[];
  }, [stepPred, ohlc, currentStep]);

  const visibleFrom = useMemo(() => {
    if (!chartCandles.length) return undefined;
    return chartCandles[0].time;
  }, [chartCandles]);

  const futureStart = useMemo(() => {
    if (!ohlc || ohlc.length <= currentStep) return undefined;
    return ohlc[currentStep]?.time;
  }, [ohlc, currentStep]);

  if (!session) return <div>Loading Simulation…</div>;

  const symbolsReady = Array.isArray(symbols) && symbols.length > 0;
  const stepLabel = session.step_labels?.[currentStep] ?? `Step ${currentStep}`;
  const isLive = !manualStep && liveDate !== null;

  return (
    <div className={styles.page}>
      {/* ── LEFT COLUMN ── */}
      <div className={styles.leftCol}>

        {/* Header row */}
        <div className={styles.header}>
          <div className={styles.titleGroup}>
            <h1>Simulation{isLive && <span className={styles.liveTag}> ● LIVE</span>}</h1>
            <span className={styles.replayBadge}>Replay: {session.replay_date}</span>
            {symbolsLoading && <span className={styles.symbolStatus}>Loading…</span>}
            {!symbolsLoading && symbolsError && (
              <span className={styles.symbolError}>Failed: {symbolsError.message}</span>
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
          <div className={styles.legend}>
            <span className={styles.legendBase}>── Base forecast (static)</span>
            <span className={styles.legendWarm}>─ ─ Warm refresh (step-updated)</span>
          </div>
        </div>

        {/* Metrics row */}
        <div className={styles.metricsRow}>
          <MetricCard label="Replay Date" value={session.replay_date} />
          <MetricCard label="Model As-Of" value={session.effective_as_of_date} />
          <MetricCard
            label="Base Return"
            value={basePred?.predicted_full_day_return != null
              ? `${basePred.predicted_full_day_return.toFixed(2)}%` : '—'}
          />
          <MetricCard
            label={`Warm (${stepLabel})`}
            value={stepPred?.predicted_full_day_return != null
              ? `${stepPred.predicted_full_day_return.toFixed(2)}%` : '—'}
          />
          <MetricCard
            label="Direction"
            value={stepPred?.predicted_direction?.toUpperCase() ?? basePred?.predicted_direction?.toUpperCase() ?? '—'}
          />
        </div>

        {/* Chart — fills remaining height */}
        <div className={styles.chartArea}>
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

      {/* ── RIGHT COLUMN: vertical slider ── */}
      <div className={styles.rightCol}>
        <div className={styles.timeHeader}>
          <span className={styles.currentTime}>{stepToMarketTime(currentStep)}</span>
          <span style={{ fontSize: '0.6rem', color: '#94a3b8' }}>ET</span>
          {manualStep && (
            <button className={styles.liveBtn} onClick={() => { setManualStep(false); setCurrentStep(computeLiveStep()); }}>
              ↺ Live
            </button>
          )}
        </div>

        <div className={styles.stepLabel}>{stepLabel}</div>

        <div className={styles.sliderTrack}>
          <span className={styles.timeTickTop}>09:30</span>

          <div className={styles.sliderGrow}>
            <input
              type="range"
              min={0}
              max={maxStepIndex}
              value={Math.min(currentStep, maxStepIndex)}
              onChange={e => { setManualStep(true); setCurrentStep(parseInt(e.target.value, 10)); }}
              className={styles.sliderVertical}
              title={stepToMarketTime(Math.min(currentStep, maxStepIndex))}
            />
          </div>

          <span className={styles.timeTickBottom}>15:45</span>
        </div>

        <div className={styles.keyboardHint}>
          <kbd>↑</kbd><kbd>↓</kbd><br />
          navigate<br />
          <kbd>L</kbd> live
        </div>
      </div>
    </div>
  );
};
