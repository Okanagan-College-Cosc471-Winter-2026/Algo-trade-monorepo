import React, { useEffect, useRef } from 'react';
import { 
  createChart, 
  ColorType, 
  IChartApi, 
  ISeriesApi, 
  UTCTimestamp 
} from 'lightweight-charts';
import styles from './CandlestickChart.module.css';

interface ChartData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface PredictionData {
  time: number;
  value: number;
}

interface CandlestickChartProps {
  data: ChartData[];
  basePrediction?: PredictionData[];
  warmPrediction?: PredictionData[];
  height?: number;
}

export const CandlestickChart: React.FC<CandlestickChartProps> = ({ 
  data, 
  basePrediction, 
  warmPrediction,
  height = 500 
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const baseSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const warmSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'white' },
        textColor: '#334155',
      },
      grid: {
        vertLines: { color: '#f1f5f9' },
        horzLines: { color: '#f1f5f9' },
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#14b8a6',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#14b8a6',
      wickDownColor: '#ef4444',
    });

    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [height]);

  useEffect(() => {
    if (candlestickSeriesRef.current && data.length > 0) {
      const formattedData = data.map(d => ({
        time: d.time as UTCTimestamp,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
      }));
      candlestickSeriesRef.current.setData(formattedData);
    }
  }, [data]);

  useEffect(() => {
    if (!chartRef.current) return;

    // Handle Base Prediction
    if (basePrediction && basePrediction.length > 0) {
      if (!baseSeriesRef.current) {
        baseSeriesRef.current = chartRef.current.addLineSeries({
          color: '#f59e0b',
          lineWidth: 2,
          lineStyle: 2, // Dotted
          title: 'Base Forecast',
        });
      }
      baseSeriesRef.current.setData(basePrediction.map(p => ({
        time: p.time as UTCTimestamp,
        value: p.value,
      })));
    } else if (baseSeriesRef.current) {
      chartRef.current.removeSeries(baseSeriesRef.current);
      baseSeriesRef.current = null;
    }

    // Handle Warm Prediction
    if (warmPrediction && warmPrediction.length > 0) {
      if (!warmSeriesRef.current) {
        warmSeriesRef.current = chartRef.current.addLineSeries({
          color: '#3b82f6',
          lineWidth: 2,
          lineStyle: 1, // Dashed
          title: 'Warm Forecast',
        });
      }
      warmSeriesRef.current.setData(warmPrediction.map(p => ({
        time: p.time as UTCTimestamp,
        value: p.value,
      })));
    } else if (warmSeriesRef.current) {
      chartRef.current.removeSeries(warmSeriesRef.current);
      warmSeriesRef.current = null;
    }
  }, [basePrediction, warmPrediction]);

  return <div ref={chartContainerRef} className={styles.container} />;
};
