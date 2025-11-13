import { FormEvent, useMemo, useState } from "react";

type MessageRole = "user" | "assistant" | "system";

interface Message {
  id: string;
  role: MessageRole;
  text: string;
}

interface ChatResponse {
  answer: string;
  sources: string[];
}

const API_URL = import.meta.env.VITE_API_URL;

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const disabled = useMemo(() => loading || !input.trim(), [loading, input]);

  async function sendMessage(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    if (!input.trim()) return;
    setError(null);

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: input.trim(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: userMessage.id, message: userMessage.text }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        const detail = payload?.detail ?? response.statusText;
        throw new Error(detail);
      }

      const data = (await response.json()) as ChatResponse;
      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: data.answer ?? "I’m sorry, I couldn’t find that information.",
      };
      setMessages((prev) => [...prev, assistantMessage]);

      if (data.sources?.length) {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "system",
            text: `Sources:\n${data.sources.map((src) => `• ${src}`).join("\n")}`,
          },
        ]);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "system",
          text: `Request failed: ${message}`,
        },
      ]);
    } finally {
      setInput("");
      setLoading(false);
    }
  }

  return (
    <section className="chat">
      <div className="messages" role="log" aria-live="polite">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            {msg.text.split("\n").map((line, index) => (
              <p key={index}>{line}</p>
            ))}
          </div>
        ))}
        {error && <div className="message error">{error}</div>}
      </div>
      <form className="composer" onSubmit={sendMessage}>
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask about snow removal, permits, or department services"
          rows={4}
          required
        />
        <button type="submit" disabled={disabled}>
          {loading ? "Sending" : "Send"}
        </button>
      </form>
    </section>
  );
}
