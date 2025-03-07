import { useState } from 'react'
import './App.css'
import Home from './Components/Home/Home'
import Login from './Components/Login/Login'
import StockAnalysis from './Components/Stock-Analysis/stock'
import LiveCandle from './Components/Algorithmic-Trading/LiveCandle'
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
const App = () => {
  return (
    <Router>
      <Routes>
        {/* <Route path="/" element={<Login />} /> */}
        <Route path="/Home" element={<Home />} />
        <Route path="Home/stock-analysis" element={<StockAnalysis />} />
        <Route path="Home/algorithmic-trading" element={<LiveCandle />} />
      </Routes>
    </Router>
  );
}

export default App
