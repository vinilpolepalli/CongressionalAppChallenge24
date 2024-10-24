// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBhUhc87Skj14MGVxa2tcvQY1zuS-RuD3w",
  authDomain: "bite-ai-73090.firebaseapp.com",
  projectId: "bite-ai-73090",
  storageBucket: "bite-ai-73090.appspot.com",
  messagingSenderId: "292961593021",
  appId: "1:292961593021:web:4e89c63cc9976ee1ddb9d4",
  measurementId: "G-53H3PW824R"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

const signInWithGoogle = () => {
  signInWithPopup(auth, provider)
    .then((result) => {
      // Handle successful login
      console.log(result.user);
    })
    .catch((error) => {
      // Handle Errors here.
      console.error(error);
    });
};

export { auth, signInWithGoogle };