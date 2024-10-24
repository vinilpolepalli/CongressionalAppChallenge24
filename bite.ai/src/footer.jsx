import React from 'react';
import './Footer.css'; // Ensure this import is correct
import Logo from '/src/assets/logo-black.png'; // Updated path to your logo
import FormImage from '/src/assets/form.png'; // Path to the form image

export default function Footer() {
    return (
        <div className="footer-container">
            <h1>Contact us with any inquiries you have!</h1>
            <hr className="footer-line" />
            <img src={Logo} alt="Logo" className="footer-logo" />
            <a href="https://forms.gle/hSSxDcpJSU9Qxgmi6" target="_blank" rel="noopener noreferrer">
                <img src={FormImage} alt="Google Form" className="form-image" />
            </a>
        </div>
    );
}