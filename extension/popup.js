chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'showSentimentResults') {
    const container = document.getElementById('tweetsContainer');
    container.innerHTML = ''; // Clear existing content

    // Add CSS dynamically
    const style = document.createElement('style');
    style.innerHTML = `
      .sentiment {
        background-color: black;
        color: white;
        border: 2px solid white;
        padding: 5px;
        margin-top: 5px;
        display: inline-block;
      }
      .positive {
        background-color: green;
        border-color: green;
      }
      .negative {
        background-color: red;
        border-color: red;
      }
      .neutral {
        background-color: gray;
        border-color: gray;
      }
    `;
    document.head.appendChild(style); // Append styles to head

     setTimeout(() => {
      request.tweets.forEach(tweet => {
        const tweetElement = document.createElement('div');
        tweetElement.classList.add('tweet');
        tweetElement.innerHTML = `<p>${tweet.text}</p>`;

        const sentimentClass = tweet.sentiment.toLowerCase() === 'positive' ? 'positive' :
                               tweet.sentiment.toLowerCase() === 'negative' ? 'negative' : 'neutral';
        tweetElement.innerHTML += `<div class="sentiment ${sentimentClass}">${tweet.sentiment}</div>`;

        container.appendChild(tweetElement);
      });
    }, 1000);
  }
});
