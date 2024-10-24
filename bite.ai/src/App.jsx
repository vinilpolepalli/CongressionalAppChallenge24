import React from 'react';
import Navbar from './Navbar';
import Footer from './Footer'; // Import the Footer component

import Analyze from './pages/Analyze';
import About from './pages/AboutUs';

import Learn from './pages/Learn';
import LandingPage from './pages/LandingPage'; // Import the LandingPage component

import { Route, Routes } from 'react-router-dom';

function App() {
  return (
    <>
      <Navbar />
      <div className="container">
        <Routes>
          <Route path="/" element={<LandingPage />} /> {/* Add the LandingPage route */}
          <Route path="/about-us" element={<About />} />

          <Route path="/analyze" element={<Analyze />} />
          <Route path="/learn" element={<Learn />} />
=
        </Routes>
      </div>
      <Footer /> {/* Add the Footer component */}
    </>
  );
}

export default App;