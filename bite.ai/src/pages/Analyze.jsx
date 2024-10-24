import React, { useState, useRef, useEffect } from 'react';
import './Analyze.css'; // Import the CSS file
import mosquitoAnalysisImage from '/src/assets/Mosquito-analysis.png'; // Import the image

export default function Analyze() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [description, setDescription] = useState('');
  const [customDescription, setCustomDescription] = useState(''); // State for custom description
  const [isDragging, setIsDragging] = useState(false);
  const [isDropped, setIsDropped] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setIsDropped(false); // Reset isDropped when file is uploaded via input
    handleSubmit(e.target.files[0]); // Automatically submit the file for prediction
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    setFile(droppedFile);
    setIsDropped(true); // Set isDropped to true when file is dropped
    handleSubmit(droppedFile); // Automatically submit the file for prediction
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleSubmit = async (fileToUpload = file) => {
    if (!fileToUpload) return;

    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      setIsLoading(true);
      setProgress(0);
      console.log('Sending request to server...');
      const response = await fetch('http://127.0.0.1:5001/predict', {
        method: 'POST',
        body: formData,
      });

      // Simulate progress
      const interval = setInterval(() => {
        setProgress((prevProgress) => {
          if (prevProgress >= 100) {
            clearInterval(interval);
            setIsLoading(false);
            return 100;
          }
          return prevProgress + 10;
        });
      }, 300);

      const data = await response.json();
      console.log('Received response from server:', data);
      if (response.ok) {
        setPrediction(data.class);
        setDescription(data.description);
        setCustomDescription(data.custom_description); // Set custom description
      } else {
        setPrediction(`Error: ${data.error}`);
        setDescription('');
        setCustomDescription('');
      }
    } catch (error) {
      console.error('Error occurred:', error);
      setPrediction(`Error: ${error.message}`);
      setDescription('');
      setCustomDescription('');
      setIsLoading(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div
      className={`analyze-container ${isDragging ? 'dragging' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      {isLoading ? (
        <div className="loading-container">
          <div className="loading-text">Analyzing bug bite...</div>
          <div className="progress-bar">
            <div className="progress" style={{ width: `${progress}%` }}></div>
          </div>
          <div className="progress-text">{progress}%</div>
        </div>
      ) : file ? (
        <div className="result-container">
          <h2 className="result-header">Using the image you uploaded, our algorithm has detected what your bug bite is.</h2>
          <div className="card-container">
            <div className="uploaded-image-card card">
              <img src={URL.createObjectURL(file)} alt="Uploaded" className="uploaded-image" />
            </div>
            <div className="prediction-card card">
              <h2>Prediction</h2>
              <p>{prediction ? `The model predicts this is a ${prediction}` : 'Loading prediction...'}</p>
              {description && (
                <>
                  <h2>Brief Bite Overview</h2>
                  <p>{description}</p>
                </>
              )}
            </div>
          </div>
          <div className="additional-cards-container">
            <div className="additional-card">
              <h2>Bite.ai™ Analysis</h2>
              <p>{customDescription ? customDescription : 'Placeholder content for Bite.ai analysis.'}</p>
            </div>
            <div className="additional-card">
              <h2>Bite.ai™ Guide</h2>
              <p>
                <a href="https://docs.google.com/document/d/e/2PACX-1vROrJSj57N6Tf6LCf7ZNx7Sfa1ICpSO7Fsi4_FvWE3L7mw1eWViBdg-FuExMiZ1PCtlrFRqhY6BnuwM/pub" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'underline' }}>
                  Link to Comprehensive Guide to Insect Bites, Stings, and Skin Conditions
                </a>
              </p>
            </div>
          </div>
        </div>
      ) : (
        <>
          {isDragging && (
            <div className="overlay">
              <p>Drop image anywhere</p>
            </div>
          )}
          <div className="dashed-box">
            <img src={mosquitoAnalysisImage} alt="Mosquito Analysis" className="analysis-image" />
            <h1>Upload an Image to Analyze the Bug Bite</h1>
            <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }}>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                style={{ display: 'none' }} // Hide the default file input
              />
              <button type="button" className="upload-button" onClick={handleUploadClick}>
                Upload Image
              </button>
              {file && <p className="upload-message">Uploaded image</p>}
            </form>
          </div>
        </>
      )}
    </div>
  );
}