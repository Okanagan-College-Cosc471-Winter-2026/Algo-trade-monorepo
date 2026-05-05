import React from 'react';
import styles from './DataTable.module.css';
import { clsx } from 'clsx';

export interface Column<T> {
  header: string;
  accessor: keyof T | ((item: T) => React.ReactNode);
  width?: string;
  className?: string;
  align?: 'left' | 'center' | 'right';
}

interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  height?: string | number;
  className?: string;
  onRowClick?: (item: T) => void;
}

export function DataTable<T>({ 
  data, 
  columns, 
  height, 
  className,
  onRowClick 
}: DataTableProps<T>) {
  return (
    <div className={clsx(styles.wrapper, className)} style={{ height }}>
      <table className={styles.table}>
        <thead className={styles.thead}>
          <tr>
            {columns.map((col, i) => (
              <th 
                key={i} 
                className={clsx(styles.th, col.align && styles[col.align])}
                style={{ width: col.width }}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className={styles.tbody}>
          {data.length > 0 ? (
            data.map((item, rowIndex) => (
              <tr 
                key={rowIndex} 
                className={clsx(styles.tr, onRowClick && styles.clickable)}
                onClick={() => onRowClick?.(item)}
              >
                {columns.map((col, colIndex) => (
                  <td 
                    key={colIndex} 
                    className={clsx(styles.td, col.align && styles[col.align], col.className)}
                  >
                    {typeof col.accessor === 'function' 
                      ? col.accessor(item) 
                      : (item[col.accessor] as React.ReactNode)}
                  </td>
                ))}
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan={columns.length} className={styles.empty}>
                No data available.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
