import React from 'react';
import { Link } from 'react-router-dom';
import './LandingPage.css'; // Import the CSS file
import aiIcon from '../assets/ai-icon.png'; // Import the AI icon image
import guideIcon from '../assets/guide.png'; // Import the AI icon image

const LandingPage = () => {
  return (
    <>
      <main className="landing-page">
        <h1 aria-label="Detect, Diagnose, Defend—Your Bite Expert">
          Detect, Diagnose, Defend — Your Bite  
          <span className="gradient-text"> Expert</span>
        </h1>
        <p>Upload an image of your bite, and we'll analyze it to 
          accurately identify the source.</p>
        <div className="button-container">
          <Link to="/analyze">
            <button className="analyze-button">Let's Go</button>
          </Link>
        </div>
      </main>
      <div className="new-container">
        <h2>Need to analyze your bug bite instantly?</h2>
        <div className="text-with-icon">
          <p>Join over 1000+ students, patients, and doctors who've
            used our free AI diagnosis tool and our bite guide
            to enhance bug bite safety.</p>
          <div className="icons-container">
            <div className="icon-wrapper">
              <div className="icon-box">
                <img src={aiIcon} alt="AI Icon" className="ai-icon" />
              </div>
              <div className="icon-text">AI Diagnosis Tool</div>
              <p className="icon-description">We designed a highly <br></br>accurate tool to diagnose bug bites </p>
            </div>
            <div className="vertical-line"></div> 
            <div className="icon-wrapper">
              <div className="icon-box">
                <img src={guideIcon} alt="Guide Icon" className="guide-icon" />
              </div>
              <div className="icon-text">Bite Guide</div>
              <p className="icon-description">We developed a highly <br></br> in-depth guide to inform<br></br> students and patients of bug bites </p>
            </div>
          </div>
        </div>
      </div>
      <div className="offerings-container">
        <h2>Our Offerings</h2>
        <p className="offerings-description">
        Empowering patients, doctors, and students, Bite.ai 
        offers free AI-powered bite detection technology and 
        a comprehensive guide to identifying different bug 
        bites. Whether you're looking to identify an unknown 
        bite or looking to deepen your knowledge on bug bites,
         our resources are designed to support your needs. 
        </p>
        <div className="card">
          
          <div className="card-content">
            <h3>Bite.ai™ Deep Learning Model</h3>
            <p>Meticulously engineered by Henry Cantor and Vinil Polepalli, our machine learning model boasts the highest accuracy on the bite-detection market, an accuracy of 91.3%!
            </p>
            <ol>
              <li>•   Classifies bug bite images by distinguishing types based on visual features.</li>
              <li>•   Uses multiple convolutional layers to automatically extract relevant patterns.</li>
              <li>•   Applies max pooling to reduce dimensionality and emphasize salient features.</li>
              <li>•   Utilizes ReLU activation for improved learning of complex patterns.</li>
              <li>•   Trained on a diverse dataset with techniques like rotation and scaling for robustness.</li>
            </ol>

          </div>
          <img src="src/assets/mosquito-bite.png" alt="Mosquito Bite" className="card-image" />
        </div>
        <div className="card">
        <img src="src/assets/ai-guide.png" alt="Bite Guide" className="card-image" />
          <div className="card-content">
            <h3>Bite.ai™ Guide</h3>
            <p>A product of hours of research and talking to various doctors and professors, we have developed our comprehensive 10+ page guide.</p>
            <ol>
              <li>•   Details on bug bite identification, symptoms, and risks.</li>
              <li>•   Practical advice for avoiding bug bites in various environments.</li>
              <li>•   Effective treatments and home remedies for symptoms.</li>
              <li>•   Images to assist in accurately identifying bug bites.</li>
              <li>•   Easy navigation for quick access to information.</li>
            </ol>
          </div>
        </div>
      </div>
    </>
  );
};

export default LandingPage;