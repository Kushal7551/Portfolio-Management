import React, { useEffect, useState } from 'react';
import Highcharts from 'highcharts/highstock';
import Navbar from '../Navbar';
import Header from '../Header';
import SideBar from '../Home/SideBar';
import './LiveCandle.css';

const LiveCandle = () => {
  const [csvData, setCsvData] = useState([]);
  const [chart, setChart] = useState(null);
  const [csvFiles, setCsvFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState('ADANIPOWER.csv');
  const [currentIndex, setCurrentIndex] = useState(0); // Track the current index of data to be added
  const [isModelRunning, setIsModelRunning] = useState(false); // Control when to display data
  const [netWorth, setNetWorth] = useState(null);
  const [balance, setBalance] = useState(null);

  useEffect(() => {
    const fetchCSVList = async () => {
      try {
        setCsvFiles(['ADANIPOWER.csv', 'BLUEDART.csv', 'NHPC.csv']);
      } catch (error) {
        console.error('Error fetching CSV file list:', error);
      }
    };
    fetchCSVList();
  }, []);

  useEffect(() => {
    const fetchCSVData = async () => {
      try {
        const response = await fetch(`http://192.168.58.2:30007/read-csv`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ fileName: selectedFile }), // Send selected file name to backend
        });

        if (!response.ok) {
          throw new Error(`Error fetching data for ${selectedFile}`);
        }

        const text = await response.json();
        console.log(text);

        const formattedData = text.map(row => ({
          x: new Date(row.Date).getTime(), // Timestamp
          open: parseFloat(row.open),
          high: parseFloat(row.high),
          low: parseFloat(row.low),
          close: parseFloat(row.close),
          signal: row.signal, // "buy" or "sell"
          quantity: row.quantity,
          netWorth: parseFloat(row.netWorth),
          balance: parseFloat(row.balance),
        }));

        setCsvData(formattedData);
        setCurrentIndex(0); // Reset the index when new file is loaded
      } catch (error) {
        console.error('Error fetching or parsing CSV:', error);
      }
    };

    fetchCSVData();
  }, [selectedFile]);

  useEffect(() => {
    if (csvData.length === 0 || chart) return;

    const newChart = Highcharts.stockChart('container', {
      chart: {
        type: 'candlestick',
        backgroundColor: '#f4f6f8',
      },
      rangeSelector: {
        selected: 1,
      },
      navigator: {
        enabled: true,
      },
      scrollbar: {
        enabled: true,
      },
      series: [
        {
          type: 'candlestick',
          name: 'Stock Price',
          data: [],
          color: '#FF0000', // Red for decreasing
          upColor: '#008000', // Green for increasing
          dataGrouping: { enabled: false },
        },
        {
          type: 'scatter',
          name: 'Buy Signal',
          data: [],
          marker: {
            symbol: 'triangle',
            radius: 8, // Bigger size for the marker
            fillColor: '#FF0000', // Red for buy
          },
          tooltip: {
            pointFormat: 'Buy Signal: <br/>Price: {point.y}',
          },
        },
        {
          type: 'scatter',
          name: 'Sell Signal',
          data: [],
          marker: {
            symbol: 'triangle-down',
            radius: 8, // Bigger size for the marker
            fillColor: '#008000', // Green for sell
          },
          tooltip: {
            pointFormat: 'Sell Signal: <br/>Price: {point.y}',
          },
        },
      ],
    });

    setChart(newChart);
  }, [csvData]);

  useEffect(() => {
    if (!isModelRunning || !chart || csvData.length === 0 || currentIndex >= csvData.length) return;

    const addDataPoint = () => {
      const nextPoint = csvData[currentIndex];
      const { x, open, high, low, close, signal, netWorth, balance } = nextPoint;

      // Add candlestick data
      chart.series[0].addPoint([x, open, high, low, close], true, false);

      // Add buy/sell signals with an offset
      if (signal === 'buy') {
        chart.series[1].addPoint({ x, y: close + 1 }, true, false); // Offset above the close price
      } else if (signal === 'sell') {
        chart.series[2].addPoint({ x, y: close - 1 }, true, false); // Offset below the close price
      }

      // Update net worth and balance
      setNetWorth(netWorth);
      setBalance(balance);

      setCurrentIndex(prevIndex => prevIndex + 1); // Move to the next data point
    };

    const intervalId = setInterval(addDataPoint, 500); // Add a new point every 0.5 seconds

    return () => clearInterval(intervalId);
  }, [isModelRunning, chart, csvData, currentIndex]);

  return (
    <div>
      <Navbar />
      <Header text="Live Candle Stick" backPath="/Home" />
      <div className="live-candle-container">
        <SideBar />
        <div className="main-content">
          <div className="dropdown-container">
            <label htmlFor="csvSelect" className="dropdown-label"> 
              Select Dataset:
            </label>
            <select
              id="csvSelect"
              value={selectedFile}
              onChange={(e) => {
                setSelectedFile(e.target.value);
                setChart(null); // Reset chart when file changes
                setIsModelRunning(false); // Reset the model state
              }}
              className="dropdown"
            >
              {csvFiles.map((file, index) => (
                <option key={index} value={file}>
                  {file}
                </option>
              ))}
            </select>
          </div>
          <button
            onClick={() => setIsModelRunning(true)}
            className="run-model-button"
          >
            Run Model
          </button>
          <div id="container" className="chart-container"></div>
          {isModelRunning && (
            <div className="stats-container">
              <h3>Model Statistics</h3>
              <p>
                <strong>Net Worth:</strong>{' '}
                {netWorth !== null ? netWorth.toFixed(2) : 'Loading...'}
              </p>
              <p>
                <strong>Cash Balance:</strong>{' '}
                {balance !== null ? balance.toFixed(2) : 'Loading...'}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveCandle;
