import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  LayoutDashboard, 
  BarChart2, 
  TrendingUp, 
  PlayCircle, 
  Database, 
  Settings,
  Activity
} from 'lucide-react';
import styles from './Sidebar.module.css';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Overview' },
  { to: '/stocks', icon: BarChart2, label: 'Stocks' },
  { to: '/predictions', icon: TrendingUp, label: 'Predictions' },
  { to: '/simulation', icon: PlayCircle, label: 'Simulation' },
  { to: '/snapshots', icon: Database, label: 'Snapshots' },
  { to: '/ops', icon: Settings, label: 'Ops' },
];

export const Sidebar: React.FC = () => {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.brand}>
        <div className={styles.brandName}>MarketSight</div>
        <div className={styles.brandSubtitle}>
          Market data, inference & dataset snapshots.
          <br />
          Optimized for fast scanning.
        </div>
      </div>
      
      <nav className={styles.nav}>
        <ul className={styles.navList}>
          {navItems.map((item) => (
            <li key={item.to} className={styles.navItem}>
              <NavLink
                to={item.to}
                className={({ isActive }) => 
                  isActive ? `${styles.navLink} ${styles.activeNavLink}` : styles.navLink
                }
              >
                <item.icon size={18} />
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      
      <div className={styles.footer}>
        <span>{new Date().toISOString().slice(0, 16).replace('T', ' ')} UTC</span>
        <Activity size={14} className={styles.activityIcon} />
      </div>
    </aside>
  );
};
