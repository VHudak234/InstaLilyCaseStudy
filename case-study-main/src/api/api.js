// Talks to the FastAPI backend running locally on :8000.
// We send the *full* conversation history each turn so the agent has context;
// the backend is stateless, which keeps things simple and easy to scale later.

const BACKEND_URL = "http://localhost:8000/chat";

export const getAIMessage = async (history) => {
  let res;
  try {
    res = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
  } catch (_networkErr) {
    // fetch() only throws on real network/CORS failures. Distinguishing this
    // from a backend-level error lets us show an accurate message.
    return {
      role: "assistant",
      content:
        "Sorry — I couldn't reach the backend. Make sure the FastAPI server is running on port 8000.",
    };
  }

  // Try to read the body as JSON regardless of status: the backend returns a
  // friendly reply (with {error: {...}}) even when Gemini is overloaded.
  let data = null;
  try {
    data = await res.json();
  } catch (_parseErr) {
    // Non-JSON response (e.g. FastAPI 500 HTML). Fall through.
  }

  if (data && data.reply) {
    return {
      role: "assistant",
      content: data.reply,
      trace: data.trace,
      error: data.error, // surfaces transient upstream issues to the UI if needed
    };
  }

  return {
    role: "assistant",
    content:
      "Sorry — something went wrong on the server. Please try again in a moment.",
  };
};
