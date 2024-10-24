import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom'; // Import useNavigate hook
import './Login.css'; // Import the CSS file for styling

export default function Login() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignup, setIsSignup] = useState(false);
  const [message, setMessage] = useState('');
  const [loggedInUser, setLoggedInUser] = useState('');
  const navigate = useNavigate(); // Initialize useNavigate

  useEffect(() => {
    const storedUser = localStorage.getItem('loggedInUser');
    if (storedUser) {
      setLoggedInUser(storedUser);
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const url = isSignup ? 'http://127.0.0.1:5001/register' : 'http://127.0.0.1:5001/login';
    const body = isSignup ? { name, email, password } : { email, password };
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    setMessage(data.message);

    if (isSignup && response.status === 201) {
      // If signup is successful, switch to login form
      setIsSignup(false);
      setEmail('');
      setPassword('');
    } else if (!isSignup && response.status === 200) {
      // If login is successful, set the logged-in user's name and navigate to symptom tracking page
      setLoggedInUser(data.name);
      localStorage.setItem('loggedInUser', data.name);
      navigate('/symptom-tracking');
    }
  };

  const handleLogout = () => {
    setLoggedInUser('');
    localStorage.removeItem('loggedInUser');
    navigate('/login');
  };

  return (
    <div className="login-container">
      {loggedInUser ? (
        <>
          <h1>Welcome, {loggedInUser}</h1>
          <button onClick={handleLogout} className="logout-button">Logout</button>
        </>
      ) : (
        <>
          <h1>{isSignup ? 'Sign Up' : 'Login'}</h1>
          <form onSubmit={handleSubmit}>
            {isSignup && (
              <div className="form-group">
                <label htmlFor="name">Name:</label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
            )}
            <div className="form-group">
              <label htmlFor="email">Email:</label>
              <input
                type="email"
                id="email"
                name="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label htmlFor="password">Password:</label>
              <input
                type="password"
                id="password"
                name="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <button type="submit" className="login-button">
              {isSignup ? 'Sign Up' : 'Login'}
            </button>
          </form>
          <p>{message}</p>
          <p>
            {isSignup ? 'Already have an account?' : "Don't have an account?"}{' '}
            <a href="#" onClick={() => setIsSignup(!isSignup)}>
              {isSignup ? 'Login' : 'Sign up'}
            </a>
          </p>
        </>
      )}
    </div>
  );
}