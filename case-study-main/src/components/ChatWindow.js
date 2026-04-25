import React, { useState, useEffect, useRef } from "react";
import "./ChatWindow.css";
import { getAIMessage } from "../api/api";
import { marked } from "marked";

// Pull part records out of the tool-call trace so we can render product cards.
// We look at the three tools that return part payloads: search_parts and
// troubleshoot return lists; get_part_details returns a single part.
const extractCards = (trace) => {
  if (!trace || trace.length === 0) return [];
  const byPn = new Map();
  for (const step of trace) {
    const r = step.result || {};
    const bucket =
      r.results /* search_parts */ ||
      r.likely_parts /* troubleshoot */ ||
      r.suggestions /* get_part_details miss with fuzzy suggestions */ ||
      r.items /* view_cart — cart lines */ ||
      (r.part_number && r.image_url ? [r] : []); /* get_part_details / get_installation_guide hit — require image_url so confirmation/compatibility payloads don't render as empty cards */
    for (const p of bucket) {
      if (p && p.part_number && !byPn.has(p.part_number)) {
        byPn.set(p.part_number, p);
      }
    }
  }
  // Cap at 6 so a chatty tool run doesn't flood the UI.
  return Array.from(byPn.values()).slice(0, 6);
};

// Human labels for the tools. Keeps the UI approachable — customers don't need
// to see implementation identifiers like "get_part_details".
const TOOL_LABELS = {
  search_parts: "Searching parts",
  get_part_details: "Looking up part",
  check_compatibility: "Checking compatibility",
  get_installation_guide: "Fetching install guide",
  troubleshoot: "Diagnosing symptom",
  add_to_cart: "Adding to cart",
  view_cart: "Showing cart",
  remove_from_cart: "Removing from cart",
  update_cart_quantity: "Updating quantity",
  initiate_checkout: "Starting checkout",
  remember_appliance: "Saving appliance",
};

// Short human label for the "key" arg of each tool — what best summarises the
// call at a glance. Falls back to the first arg value otherwise.
const summariseArgs = (args) => {
  if (!args) return "";
  const priority = ["part_number", "symptom", "query", "model_number", "appliance_type"];
  for (const k of priority) {
    if (args[k]) return String(args[k]);
  }
  const first = Object.values(args)[0];
  return first ? String(first) : "";
};

const truncate = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

const TraceView = ({ trace }) => {
  if (!trace || trace.length === 0) return null;
  return (
    <div className="trace">
      <div className="trace-pills">
        {trace.map((step, i) => (
          <span key={i} className="trace-pill">
            <span className="trace-pill-check">✓</span>
            <span className="trace-pill-tool">{TOOL_LABELS[step.tool] || step.tool}</span>
            {summariseArgs(step.args) && (
              <span className="trace-pill-arg">{truncate(summariseArgs(step.args), 40)}</span>
            )}
          </span>
        ))}
      </div>
    </div>
  );
};

const ProductCard = ({ part, onSelect, disabled }) => (
  // Card body is a button: clicking selects the part (sends a chat message).
  // The corner arrow is a separate link out to PartSelect, so a customer
  // can either continue the conversation with the agent or jump to the
  // source page — but clicking the body never accidentally navigates away.
  <div className="product-card">
    <button
      type="button"
      className="product-card-button"
      onClick={() => onSelect(part)}
      disabled={disabled}
      aria-label={`Ask about ${part.name}, part number ${part.part_number}`}
    >
      {part.image_url && (
        <img
          className="product-card-image"
          src={part.image_url}
          alt={part.name}
          onError={(e) => {
            e.target.style.display = "none";
          }}
        />
      )}
      <div className="product-card-body">
        <div className="product-card-name">{part.name}</div>
        <div className="product-card-meta">
          {part.brand && <span>{part.brand}</span>}
          <span className="product-card-pn">{part.part_number}</span>
        </div>
        {typeof part.price === "number" && part.price > 0 && (
          <div className="product-card-price">${part.price.toFixed(2)}</div>
        )}
      </div>
    </button>
    {part.url && (
      <a
        className="product-card-external"
        href={part.url}
        target="_blank"
        rel="noopener noreferrer"
        title="View on PartSelect"
        aria-label="View on PartSelect"
        onClick={(e) => e.stopPropagation()}
      >
        ↗
      </a>
    )}
  </div>
);

function ChatWindow() {

  const defaultMessage = [{
    role: "assistant",
    content: "Hi, how can I help you today?"
  }];

  const [messages,setMessages] = useState(defaultMessage)
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
      scrollToBottom();
  }, [messages, isLoading]);

  const handleSelectPart = (part) => {
    // Card click: ask the agent about this specific part. Cheap, chat-native
    // way to "select from a list" without making the user retype a part number.
    handleSend(`Tell me more about ${part.part_number}`);
  };

  const handleSend = async (input) => {
    if (input.trim() === "" || isLoading) return;

    const userMessage = { role: "user", content: input };
    // The agent needs the full conversation history (it's stateless on the backend).
    const history = [...messages, userMessage];
    setMessages(history);
    setInput("");
    setIsLoading(true);

    const newMessage = await getAIMessage(history);
    setMessages(prev => [...prev, newMessage]);
    setIsLoading(false);
  };

  return (
      <div className="messages-container">
          {messages.map((message, index) => {
              const cards = message.role === "assistant" ? extractCards(message.trace) : [];
              return (
                <div key={index} className={`${message.role}-message-container`}>
                    {message.content && (
                        <div className={`message ${message.role}-message`}>
                            {message.role === "user" ? (
                                // User input is plain text; rendering it through marked()
                                // adds a trailing newline that white-space: pre-line then
                                // honours, forcing every short message onto two lines.
                                <span className="message-text">{message.content}</span>
                            ) : (
                                <div
                                    className="message-text"
                                    dangerouslySetInnerHTML={{
                                        __html: marked(message.content)
                                            .replace(/<p>|<\/p>/g, "")
                                            .trim(),
                                    }}
                                />
                            )}
                        </div>
                    )}
                    {message.role === "assistant" && <TraceView trace={message.trace} />}
                    {cards.length > 0 && (
                        <div className="product-cards">
                            {cards.map((p) => (
                                <ProductCard
                                    key={p.part_number}
                                    part={p}
                                    onSelect={handleSelectPart}
                                    disabled={isLoading}
                                />
                            ))}
                        </div>
                    )}
                </div>
              );
          })}
          {isLoading && (
            <div className="assistant-message-container">
              <div className="message assistant-message thinking">
                <span className="thinking-dot" />
                <span className="thinking-text">Thinking…</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
          <div className="input-area">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type a message..."
              disabled={isLoading}
              onKeyPress={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  handleSend(input);
                  e.preventDefault();
                }
              }}
              rows="3"
            />
            <button className="send-button" onClick={() => handleSend(input)} disabled={isLoading}>
              Send
            </button>
          </div>
      </div>
);
}

export default ChatWindow;
