import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [recognizedText, setRecognizedText] = useState("");

  useEffect(() => {
    // Poll the backend every 1 second for the recognized text
    const interval = setInterval(() => {
      fetch("/recognized_text")
        .then(response => response.json())
        .then(data => {
          setRecognizedText(data.recognized_text);
        })
        .catch(error => console.error("Error fetching recognized text:", error));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Function to clear the recognized text by calling the backend endpoint
  const handleClear = () => {
    fetch("/clear_text", {
      method: "POST",
    })
      .then(response => response.json())
      .then(data => {
        setRecognizedText(data.recognized_text);
      })
      .catch(error => console.error("Error clearing recognized text:", error));
  };

  return (
    <div>
      <header>
        <h1>ASL Recognition</h1>
      </header>
      <div className="video-container">
        <img src="/video_feed" alt="Video Feed" />
      </div>
      <div className="recognized-text">
        <h2>Detected Text:</h2>
        <p>{recognizedText}</p>
        <button className="clear-btn" onClick={handleClear}>
          Clear
        </button>
      </div>
    </div>
  );
}

export default App;
