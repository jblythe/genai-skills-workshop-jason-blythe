import { Chat } from "./components/Chat";

function App() {
  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>Alaska Department of Snow Virtual Assistant</h1>
        <p>Ask about snow removal, permits, parking bans, and department services.</p>
      </header>
      <Chat />
      <footer className="app-footer">
        <small>Responses generated from ADS FAQs via secure Vertex AI integration.</small>
      </footer>
    </main>
  );
}

export default App;
