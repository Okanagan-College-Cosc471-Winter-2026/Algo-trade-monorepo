import React, { useEffect, useRef } from 'react';
import styles from './LogViewer.module.css';

interface LogViewerProps {
  lines: string[];
  height?: string | number;
  autoScroll?: boolean;
}

export const LogViewer: React.FC<LogViewerProps> = ({ 
  lines, 
  height = 420, 
  autoScroll = true 
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [lines, autoScroll]);

  const getLineColor = (line: string) => {
    const ll = line.toLowerCase();
    if (ll.includes('[error]') || ll.includes('failed')) return '#f87171';
    if (ll.includes('[warning]') || ll.includes('warning') || ll.includes('warn')) return '#fbbf24';
    if (ll.includes('pipeline complete') || ll.includes('] ok') || ll.includes('success')) return '#4ade80';
    if (ll.includes('=====')) return '#60a5fa';
    return '#cbd5e1';
  };

  return (
    <div 
      className={styles.container} 
      style={{ height }}
      ref={scrollRef}
    >
      {lines.length > 0 ? (
        lines.map((line, i) => (
          <div 
            key={i} 
            className={styles.line}
            style={{ color: getLineColor(line) }}
          >
            {line}
          </div>
        ))
      ) : (
        <div className={styles.empty}>No logs available.</div>
      )}
    </div>
  );
};
