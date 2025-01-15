chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyzeText") {
    fetch("http://127.0.0.1:5000/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: message.text })
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Response from Flask:", data);
        sendResponse({ sentiment: data.sentiment, sarcasm: data.sarcasm });
      })
      .catch((error) => {
        console.error("Error:", error);
        sendResponse({ error: "Error analyzing text" });
      });

    return true; // Keep the message channel open for async response
  }
});
