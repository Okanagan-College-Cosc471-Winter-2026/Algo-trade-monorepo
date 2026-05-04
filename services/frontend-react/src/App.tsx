import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Overview, Stocks, Predictions, Simulation, Snapshots, Ops } from './pages';
import './App.css';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/stocks" element={<Stocks />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/simulation" element={<Simulation />} />
          <Route path="/snapshots" element={<Snapshots />} />
          <Route path="/ops" element={<Ops />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
