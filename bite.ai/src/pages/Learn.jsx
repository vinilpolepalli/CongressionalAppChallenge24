import React, { useState } from 'react';
import './AboutUs.css'; // Ensure this import is correct
import './Learn.css';
import bugBiteLearn from '../assets/bug-bite-learn.png'; // Adjust the path as needed

export default function About() {
    const [activeIndex, setActiveIndex] = useState(null);

    const toggleAccordion = (index) => {
        setActiveIndex(activeIndex === index ? null : index);
    };

    return (
        <div className="about-page">
            <div className="about-container">
                <h1>Importance of bite detection</h1>
                <div className="content-wrapper">
                    <p>
                        Bite detection refers to the identification and analysis 
                        of bites from various sources, such as insects and animals,
                        to assess potential health risks and ensure appropriate 
                        responses. Accurate bite detection is crucial as it 
                        enables timely intervention, reduces the risk of 
                        infections or allergic reactions, and helps prevent the 
                        spread of diseases associated with bites, such as Lyme 
                        disease, dengue, and malaria.
                    </p>
                    <img src={bugBiteLearn} alt="Bug Bite Learn" className="learn-image" />
                </div>
            </div>
            <div className="team-section">
                <h1>Our solution</h1>
                <div className="team-members">
                    <div className="learn-container">
                        <h2>
                            An AI model that can detect up to 
                            10 different bites at an accuracy 
                            of greater than 91%.
                        </h2>
                        <div className="faq-section">
                            <div className="accordion">
                                <div className="accordion-item">
                                    <div className="accordion-title" onClick={() => toggleAccordion(0)}>
                                        <h4>What is bite detection?</h4>
                                        <span className={`arrow ${activeIndex === 0 ? 'up' : 'down'}`}>
                                            <svg width="12" height="12" viewBox="0 0 24 24">
                                                <path d="M12 2l-10 10h20z" />
                                            </svg>
                                        </span>
                                    </div>
                                    {activeIndex === 0 && (
                                        <div className="accordion-content">
                                            <p>Bite detection refers to the identification and analysis of bites from various sources, such as insects and animals, to assess potential health risks and ensure appropriate responses.</p>
                                        </div>
                                    )}
                                </div>
                                <div className="accordion-item">
                                    <div className="accordion-title" onClick={() => toggleAccordion(1)}>
                                        <h4>How accurate is the AI model?</h4>
                                        <span className={`arrow ${activeIndex === 1 ? 'up' : 'down'}`}>
                                            <svg width="12" height="12" viewBox="0 0 24 24">
                                                <path d="M12 2l-10 10h20z" />
                                            </svg>
                                        </span>
                                    </div>
                                    {activeIndex === 1 && (
                                        <div className="accordion-content">
                                            <p>Our AI model can detect up to 10 different bites with an accuracy of greater than 91%.</p>
                                        </div>
                                    )}
                                </div>
                                <div className="accordion-item">
                                    <div className="accordion-title" onClick={() => toggleAccordion(2)}>
                                        <h4>How does the AI model work?</h4>
                                        <span className={`arrow ${activeIndex === 2 ? 'up' : 'down'}`}>
                                            <svg width="12" height="12" viewBox="0 0 24 24">
                                                <path d="M12 2l-10 10h20z" />
                                            </svg>
                                        </span>
                                    </div>
                                    {activeIndex === 2 && (
                                        <div className="accordion-content">
                                            <p>The AI model uses advanced machine learning algorithms to analyze bite images and provide accurate detection results.</p>
                                            <img src="src/assets/model-diag.png" alt="Model Diagram" className="model-diagram" />
                                        </div>
                                    )}
                                </div>
                                
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}