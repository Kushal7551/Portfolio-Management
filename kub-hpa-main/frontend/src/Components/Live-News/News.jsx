import React, { useEffect, useRef, useState } from 'react';

const News = () => {
  const [newsData, setNewsData] = useState([]);
  const [loading, setLoading] = useState(true); // State to manage loading
  const scrollRef = useRef(null);

  const fetchData = async () => {
    try {
      let dataAvailable = false;

      // Poll the backend for sentiment data
      while (!dataAvailable) {
        const response = await fetch('http://192.168.58.2:30007/sample', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        if (response.ok) {
          const data = await response.json();
          console.log('Received sentiment data:', data);

          if (data.length > 0) {
            // Process and set the data
            const processedData = data.map((row) => ({
              title: row.Headline,
              positive: parseFloat(row.Positive),
              negative: parseFloat(row.Negative),
              neutral: parseFloat(row.Neutral),
            }));
            setNewsData(processedData);
            dataAvailable = true; // Stop polling when data is available
            setLoading(false); // Hide loading screen
          } else {
            console.log('Data not available yet, retrying...');
            await new Promise((resolve) => setTimeout(resolve, 3000)); // Wait 3 seconds before retrying
          }
        } else {
          console.log('Data not available yet, retrying...');
          await new Promise((resolve) => setTimeout(resolve, 3000));
        }
      }
    } catch (error) {
      console.error('Error fetching sentiment data:', error);
      setLoading(false); // Hide loading screen even if there's an error
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (scrollRef.current) {
        const container = scrollRef.current;
        const scrollHeight = container.scrollHeight;
        const clientHeight = container.clientHeight;

        if (container.scrollTop + clientHeight < scrollHeight) {
          container.scrollTop += clientHeight / 3;
        } else {
          container.scrollTop = 0;
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="bg-gray-100 min-h-screen flex items-center justify-center p-8">
      {loading ? (
        // Enhanced loading screen
        <div className="flex flex-col items-center space-y-6">
          <div className="relative">
            <div className="animate-spin rounded-full h-24 w-24 border-t-4 border-blue-500 border-solid border-opacity-60"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-lg font-semibold text-gray-700">⏳</span>
            </div>
          </div>
          <h2 className="text-2xl font-semibold text-gray-800">Model is running...</h2>
          <p className="text-gray-600 text-center max-w-md">
            We are currently analyzing data to provide you with the latest sentiment updates. This might take a few seconds. Thank you for your patience!
          </p>
        </div>
      ) : (
        // Display news data when available
        <div className="bg-white rounded-xl shadow-lg p-6 max-w-3xl w-full">
          <h1 className="text-3xl font-bold text-gray-800 mb-6">Latest News</h1>
          <div ref={scrollRef} className="space-y-4 max-h-96 overflow-y-auto custom-scrollbar p-2">
            {newsData.map((article, index) => {
              let sentimentColor = '';
              let sentimentScore = 0;

              if (article.positive > article.negative && article.positive > article.neutral) {
                sentimentScore = article.positive;
                sentimentColor = 'text-green-600'; // Positive sentiment
              } else if (article.negative > article.positive && article.negative > article.neutral) {
                sentimentScore = article.negative;
                sentimentColor = 'text-red-600'; // Negative sentiment
              } else {
                sentimentScore = article.neutral;
                sentimentColor = 'text-yellow-500'; // Neutral sentiment
              }

              return (
                <div
                  key={index}
                  className="flex items-start bg-gray-50 p-4 rounded-lg shadow-sm hover:shadow-md transition cursor-pointer"
                >
                  <div className="w-16 h-16 bg-blue-100 flex items-center justify-center rounded-md">
                    <span className="text-xl font-bold text-gray-600">{index + 1}</span>
                  </div>
                  <div className="ml-4">
                    <h2 className={`text-lg font-semibold ${sentimentColor}`}>{article.title}</h2>
                    <p className={`mt-1 text-sm font-medium ${sentimentColor}`}>
                      Sentiment Score: {sentimentScore.toFixed(2)}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default News;
