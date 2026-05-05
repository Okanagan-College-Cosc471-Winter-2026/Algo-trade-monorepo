import React from 'react';
import styles from './ProgressBar.module.css';
import { clsx } from 'clsx';

interface ProgressBarProps {
  value: number; // 0 to 1
  label?: string;
  subLabel?: string;
  className?: string;
  variant?: 'primary' | 'success' | 'warning' | 'error';
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ 
  value, 
  label, 
  subLabel, 
  className,
  variant = 'primary' 
}) => {
  const percentage = Math.min(Math.max(value * 100, 0), 100);

  return (
    <div className={clsx(styles.container, className)}>
      {(label || subLabel) && (
        <div className={styles.header}>
          {label && <span className={styles.label}>{label}</span>}
          {subLabel && <span className={styles.subLabel}>{subLabel}</span>}
        </div>
      )}
      <div className={styles.track}>
        <div 
          className={clsx(styles.fill, styles[variant])} 
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};
