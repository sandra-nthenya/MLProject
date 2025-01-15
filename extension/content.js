function analyzeTweets() {
  console.log("Analyzing tweets..."); // Check if this logs

  document.querySelectorAll("article").forEach((tweet) => {
    if (!tweet.dataset.processed) {
      tweet.dataset.processed = true;

      const textElement = tweet.querySelector("div[lang]");
      if (textElement) {
        const text = textElement.innerText;
        console.log("Tweet Text:", text); // Log to verify the tweet text extraction

        chrome.runtime.sendMessage(
          { action: "analyzeText", text: text },
          (response) => {
            if (response && response.sentiment) {
              const resultDiv = document.createElement("div");
              resultDiv.style.border = "2px solid black";
              resultDiv.style.marginTop = "8px";
              resultDiv.style.padding = "8px";
              resultDiv.style.backgroundColor = "#222";
              resultDiv.style.color = "#fff";
              resultDiv.innerText = `Sentiment: ${response.sentiment}, Sarcasm: ${response.sarcasm}`;
              tweet.appendChild(resultDiv);
            } else {
              console.error("Error receiving response:", response.error);
            }
          }
        );
      }
    }
  });
}

const observer = new MutationObserver(() => analyzeTweets());
observer.observe(document.body, { childList: true, subtree: true });
