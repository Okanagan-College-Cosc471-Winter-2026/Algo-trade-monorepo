import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { api } from '../api/client';
import { useApi, useInterval } from '../hooks';
import { CandlestickChart, MetricCard } from '../components';
import { OHLC } from '../types';
import styles from './Stocks.module.css';

function isMarketOpen(): boolean {
  const now = new Date();
  const dow = now.toLocaleDateString('en-US', { timeZone: 'America/New_York', weekday: 'short' });
  if (dow === 'Sat' || dow === 'Sun') return false;
  const t = now.toLocaleString('en-US', {
    timeZone: 'America/New_York', hour12: false, hour: '2-digit', minute: '2-digit',
  });
  const [h, m] = t.split(':').map(Number);
  const mins = h * 60 + m;
  return mins >= 9 * 60 + 30 && mins < 16 * 60;
}

function computeLiveStep(): number {
  const now = new Date();
  const t = now.toLocaleString('en-US', {
    timeZone: 'America/New_York', hour12: false, hour: '2-digit', minute: '2-digit',
  });
  const [h, m] = t.split(':').map(Number);
  const mins = h * 60 + m;
  const open = 9 * 60 + 30;
  if (mins < open) return 0;
  if (mins >= 15 * 60 + 45) return 25;
  return Math.min(25, Math.floor((mins - open) / 15));
}

function todayET(): string {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

type Range = '1D' | '7D' | '30D' | '90D' | '180D' | '365D';
const RANGE_DAYS: Record<Range, number> = { '1D': 0, '7D': 7, '30D': 30, '90D': 90, '180D': 180, '365D': 365 };

export const Stocks: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [range, setRange] = useState<Range>('1D');
  const [showPredictions, setShowPredictions] = useState(true);
  const [refreshKey, setRefreshKey] = useState(0);
  const [lastUpdated, setLastUpdated] = useState(() => new Date());
  const [live, setLive] = useState(() => isMarketOpen());

  const intraday = range === '1D';
  const days = RANGE_DAYS[range];
  const today = todayET();
  const liveStep = computeLiveStep();

  const { data: stocks } = useApi(() => api.listStocks(), []);

  const { data: dailyOHLC, loading: dailyLoading } = useApi(
    () => !intraday ? api.getOHLC(selectedSymbol, days) : Promise.resolve(null),
    [selectedSymbol, days, intraday, refreshKey],
  );
  const { data: intradayOHLC, loading: intradayLoading } = useApi(
    () => intraday ? api.simOHLC(selectedSymbol, today) : Promise.resolve(null),
    [selectedSymbol, intraday, today, refreshKey],
  );

  const { data: simBase } = useApi(
    () => (intraday && showPredictions) ? api.simBase(selectedSymbol) : Promise.resolve(null),
    [selectedSymbol, intraday, showPredictions, refreshKey],
  );
  const { data: simStep } = useApi(
    () => (intraday && showPredictions) ? api.simStep(selectedSymbol, liveStep) : Promise.resolve(null),
    [selectedSymbol, intraday, showPredictions, liveStep, refreshKey],
  );

  const [basePred, setBasePred] = useState<any>(null);
  const [warmPred, setWarmPred] = useState<any>(null);
  useEffect(() => {
    if (!intraday && showPredictions) {
      api.predictBase(selectedSymbol).then(setBasePred).catch(() => setBasePred(null));
      api.predict(selectedSymbol).then(setWarmPred).catch(() => setWarmPred(null));
    } else {
      setBasePred(null);
      setWarmPred(null);
    }
  }, [intraday, showPredictions, selectedSymbol, refreshKey]);

  const ohlc: OHLC[] = intraday ? (intradayOHLC ?? []) : (dailyOHLC ?? []);
  const ohlcLoading = intraday ? intradayLoading : dailyLoading;

  const selectedStock = useMemo(
    () => stocks?.find(s => s.symbol === selectedSymbol),
    [stocks, selectedSymbol],
  );

  // Live status refresh
  useEffect(() => {
    const id = setInterval(() => setLive(isMarketOpen()), 60_000);
    return () => clearInterval(id);
  }, []);

  // Auto-refresh every 15 min
  const doRefresh = useCallback(() => {
    setRefreshKey(k => k + 1);
    setLastUpdated(new Date());
  }, []);
  useInterval(doRefresh, 60_000);

  // Base prediction overlay
  const formattedBasePred = useMemo(() => {
    if (intraday) {
      if (!simBase || !ohlc.length) return undefined;
      const anchor = ohlc[0].open;
      const baseTs = ohlc[0].time;
      return simBase.bars.map((b: any, i: number) => ({
        time: baseTs + i * 900,
        value: anchor * Math.exp(b.pred_log_return),
      }));
    }
    if (!basePred?.path || !ohlc.length) return undefined;
    const lastBar = ohlc[ohlc.length - 1];
    const predictedClose = basePred.path[basePred.path.length - 1]?.pred_close;
    if (!predictedClose) return undefined;
    const nextDayTs = Math.floor(new Date(basePred.prediction_date.split('T')[0]).getTime() / 1000);
    return [
      { time: lastBar.time, value: lastBar.close },
      { time: nextDayTs, value: predictedClose },
    ];
  }, [intraday, simBase, ohlc, basePred]);

  // Warm prediction overlay
  const formattedWarmPred = useMemo(() => {
    if (intraday) {
      if (!simStep || !ohlc.length) return undefined;
      const anchorIdx = Math.min(liveStep, ohlc.length - 1);
      const anchor = ohlc[anchorIdx]?.close ?? ohlc[0].open;
      const baseTs = ohlc[0].time;
      return simStep.bars
        .map((b: any, i: number) => {
          const barIdx = liveStep + i;
          if (barIdx >= 26) return null;
          return { time: baseTs + barIdx * 900, value: anchor * Math.exp(b.pred_log_return) };
        })
        .filter(Boolean) as { time: number; value: number }[];
    }
    if (!warmPred?.path || !ohlc.length) return undefined;
    const lastBar = ohlc[ohlc.length - 1];
    const predictedClose = warmPred.path[warmPred.path.length - 1]?.pred_close;
    if (!predictedClose) return undefined;
    const nextDayTs = Math.floor(new Date(warmPred.prediction_date.split('T')[0]).getTime() / 1000);
    return [
      { time: lastBar.time, value: lastBar.close },
      { time: nextDayTs, value: predictedClose },
    ];
  }, [intraday, simStep, ohlc, liveStep, warmPred]);

  const stats = useMemo(() => {
    if (!ohlc.length) return null;
    const latest = ohlc[ohlc.length - 1];
    const prev = intraday ? ohlc[0].open : (ohlc[ohlc.length - 2]?.close ?? latest.close);
    const change = latest.close - prev;
    const changePct = prev ? (change / prev) * 100 : 0;
    return {
      latest: latest.close,
      change,
      changePct,
      high: Math.max(...ohlc.map(d => d.high)),
      low: Math.min(...ohlc.map(d => d.low)),
      volume: latest.volume,
    };
  }, [ohlc, intraday]);

  const fmtTimeET = (d: Date) =>
    d.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit' });

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.titleGroup}>
          <h1>
            Stocks
            {live && intraday && <span className={styles.liveTag}>● LIVE</span>}
          </h1>
          <select
            value={selectedSymbol}
            onChange={e => setSelectedSymbol(e.target.value)}
            className={styles.symbolSelect}
          >
            {stocks?.map(s => (
              <option key={s.symbol} value={s.symbol}>{s.symbol} — {s.name}</option>
            ))}
          </select>
        </div>
        <div className={styles.controls}>
          <div className={styles.rangeButtons}>
            {(['1D', '7D', '30D', '90D', '180D', '365D'] as Range[]).map(r => (
              <button key={r} onClick={() => setRange(r)} className={range === r ? styles.activeRange : ''}>
                {r}
              </button>
            ))}
          </div>
          <label className={styles.toggle}>
            <input type="checkbox" checked={showPredictions} onChange={e => setShowPredictions(e.target.checked)} />
            Predictions
          </label>
          <button className={styles.refreshBtn} onClick={doRefresh} title="Refresh now">↻</button>
        </div>
      </header>

      {intraday && (
        <div className={styles.intradayMeta}>
          <span className={styles.metaDate}>{today} · 15-min bars</span>
          {live
            ? <span className={styles.liveDot}>● refreshes every 1 min</span>
            : <span className={styles.metaMuted}>market closed</span>
          }
          <span className={styles.updatedAt}>updated {fmtTimeET(lastUpdated)} ET</span>
        </div>
      )}

      {stats && (
        <div className={styles.metricsGrid}>
          <MetricCard
            label={intraday ? 'Current' : 'Last Close'}
            value={`$${stats.latest.toFixed(2)}`}
            delta={`${stats.change > 0 ? '+' : ''}${stats.change.toFixed(2)} (${stats.changePct.toFixed(2)}%)`}
            deltaType={stats.change >= 0 ? 'increase' : 'decrease'}
          />
          <MetricCard label={intraday ? 'Day High' : `${range} High`} value={`$${stats.high.toFixed(2)}`} />
          <MetricCard label={intraday ? 'Day Low' : `${range} Low`} value={`$${stats.low.toFixed(2)}`} />
          <MetricCard label="Volume" value={stats.volume?.toLocaleString() ?? '—'} />
        </div>
      )}

      <div className={styles.chartWrapper}>
        {ohlcLoading ? (
          <div className={styles.loading}>Loading chart…</div>
        ) : ohlc.length === 0 ? (
          <div className={styles.loading}>
            {intraday ? `No intraday data for ${today} yet` : 'No data available'}
          </div>
        ) : (
          <CandlestickChart
            data={ohlc}
            basePrediction={showPredictions ? formattedBasePred : undefined}
            warmPrediction={showPredictions ? formattedWarmPred : undefined}
          />
        )}
      </div>

      {showPredictions && intraday && (simBase || simStep) && (
        <div className={styles.predRow}>
          {simBase && (
            <div className={styles.predCard}>
              <span className={styles.predLabel}>Base forecast</span>
              <span className={`${styles.predDir} ${simBase.predicted_direction === 'up' ? styles.up : styles.down}`}>
                {simBase.predicted_direction?.toUpperCase()}
              </span>
              <span className={styles.predReturn}>
                {simBase.predicted_full_day_return != null
                  ? `${simBase.predicted_full_day_return > 0 ? '+' : ''}${(simBase.predicted_full_day_return * 100).toFixed(2)}%`
                  : '—'}
              </span>
              <span className={styles.predLegendBase} />
            </div>
          )}
          {simStep && (
            <div className={styles.predCard}>
              <span className={styles.predLabel}>Warm forecast (step {liveStep})</span>
              <span className={`${styles.predDir} ${simStep.predicted_direction === 'up' ? styles.up : styles.down}`}>
                {simStep.predicted_direction?.toUpperCase()}
              </span>
              <span className={styles.predReturn}>
                {simStep.predicted_full_day_return != null
                  ? `${simStep.predicted_full_day_return > 0 ? '+' : ''}${(simStep.predicted_full_day_return * 100).toFixed(2)}%`
                  : '—'}
              </span>
              <span className={styles.predLegendWarm} />
            </div>
          )}
        </div>
      )}

      <div className={styles.infoSection}>
        <div className={styles.card}>
          <h2>Company</h2>
          <div className={styles.infoGrid}>
            <div><strong>Sector:</strong> {selectedStock?.sector || '—'}</div>
            <div><strong>Industry:</strong> {selectedStock?.industry || '—'}</div>
            <div><strong>Exchange:</strong> {selectedStock?.exchange || '—'}</div>
          </div>
        </div>
        {!intraday && showPredictions && (basePred || warmPred) && (
          <div className={styles.card}>
            <h2>Prediction Metadata</h2>
            <div className={styles.infoGrid}>
              {basePred && <div><strong>Base:</strong> {basePred.model_version} · {basePred.prediction_date?.split('T')[0]}</div>}
              {warmPred && <div><strong>Warm:</strong> refreshed {warmPred.refreshed_at?.slice(11, 16) || '—'} UTC</div>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
