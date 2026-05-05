import React from 'react';
import styles from './MetricCard.module.css';

interface MetricCardProps {
  label: string;
  value: React.ReactNode;
  delta?: React.ReactNode;
  deltaType?: 'increase' | 'decrease' | 'neutral';
  help?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({ label, value, delta, deltaType, help }) => {
  return (
    <div className={styles.card}>
      <div className={styles.label}>
        {label}
        {help && <span className={styles.help} title={help}>ⓘ</span>}
      </div>
      <div className={styles.value}>{value}</div>
      {delta !== undefined && (
        <div className={`${styles.delta} ${deltaType ? styles[deltaType] : ''}`}>
          {delta}
        </div>
      )}
    </div>
  );
};
