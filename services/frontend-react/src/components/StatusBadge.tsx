import React from 'react';
import styles from './StatusBadge.module.css';
import { clsx } from 'clsx';

export type StatusType = 
  | 'success' | 'warning' | 'error' | 'info' | 'neutral'
  | 'running' | 'pending' | 'completed' | 'failed' | 'cancelled' | 'timeout'
  | 'online' | 'offline' | 'fresh' | 'stale';

interface StatusBadgeProps {
  status: StatusType | string;
  label?: string;
  className?: string;
}

const statusMap: Record<string, { className: string; defaultLabel: string }> = {
  success: { className: styles.success, defaultLabel: 'Success' },
  warning: { className: styles.warning, defaultLabel: 'Warning' },
  error: { className: styles.error, defaultLabel: 'Error' },
  info: { className: styles.info, defaultLabel: 'Info' },
  neutral: { className: styles.neutral, defaultLabel: 'Neutral' },
  running: { className: styles.running, defaultLabel: 'Running' },
  pending: { className: styles.pending, defaultLabel: 'Pending' },
  completed: { className: styles.success, defaultLabel: 'Completed' },
  failed: { className: styles.error, defaultLabel: 'Failed' },
  cancelled: { className: styles.neutral, defaultLabel: 'Cancelled' },
  timeout: { className: styles.error, defaultLabel: 'Timeout' },
  online: { className: styles.success, defaultLabel: 'Online' },
  offline: { className: styles.error, defaultLabel: 'Offline' },
  fresh: { className: styles.success, defaultLabel: 'Fresh' },
  stale: { className: styles.warning, defaultLabel: 'Stale' },
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, label, className }) => {
  const normalizedStatus = status.toLowerCase();
  const config = statusMap[normalizedStatus] || { className: styles.neutral, defaultLabel: status };

  return (
    <span className={clsx(styles.badge, config.className, className)}>
      {label || config.defaultLabel}
    </span>
  );
};
