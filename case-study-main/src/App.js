import React from "react";
import "./App.css";
import ChatWindow from "./components/ChatWindow";
import logo from "./PartSelect-Logo.png";

function App() {

  return (
    <div className="App">
      <header className="heading">
        <img src={logo} alt="PartSelect" className="brand-logo" />
        <span className="brand-divider" aria-hidden="true">|</span>
        <span className="brand-title">Assistant</span>
      </header>
        <ChatWindow/>
    </div>
  );
}

export default App;
