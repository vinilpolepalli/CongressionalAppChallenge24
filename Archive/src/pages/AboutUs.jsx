import React from 'react';
import './AboutUs.css'; // Ensure this import is correct
import Henry from '/src/assets/Henry.jpg';
import Vinil from '/src/assets/Vinil.png';

export default function About() {
    return (
        <div className="about-page">
            <div className="about-container">
                <h1>MISSION STATEMENT</h1>
                <p>
                    At bite.ai, our mission is to revolutionize the way people identify
                     and manage bug bites. Using cutting-edge AI technology, we aim to 
                     provide instant, reliable, and accurate detection of bug bites, 
                     empowering individuals to take control of their health and 
                     well-being.
                </p>
            </div>
            <div className="team-section">
                <h1>OUR TEAM</h1>
                <div className="team-members">
                    <div className="team-member">
                        <img src={Henry} alt="Henry" />
                        <div className="team-info">
                            <h2>Henry</h2>
                            <p>Co-Founder</p>
                        </div>
                    </div>
                    <div className="team-member">
                    <img src={Vinil} alt="Vinil" />
                        <div className="team-info">
                            <h2>Vinil</h2>
                            <p>Co-Founder</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}