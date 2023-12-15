document.addEventListener("DOMContentLoaded", function () {
    const inputText = document.getElementById("inputText");
    const resultDiv = document.getElementById("result");
    const getSentimentButton = document.getElementById("getSentimentButton");

    getSentimentButton.addEventListener("click", async function () {
        const sentence = inputText.value.trim();

        try {

            const req = new XMLHttpRequest();
            req.open("POST", "http://localhost:8000");

            var jsonData = {
                review: sentence
            };
            var jsonString = JSON.stringify(jsonData);
            console.log(jsonString)
            req.send(jsonString);

            req.onload = function() {
                if (req.status != 200) { // analyze HTTP status of the response
                alert(`Error ${req.status}: ${req.statusText}`); // e.g. 404: Not Found
                } else { // show the result
                    const data = JSON.parse(req.response)
                    console.log(data.sentiment_score)

                    threshHold = 0.0

                    let sentiment = "Negative"
                    if(data.sentiment_score > threshHold){
                        sentiment = "Positive"
                    }


                    resultDiv.textContent = "Sentiment: " + sentiment + 
                                            "\nSentiment_score: " + data.sentiment_score;
                }
            };                    
            
        } catch (error) {
            console.error("Error:", error);
        }
        
    });

    function getSentiment(sentence) {
        const req = new XMLHttpRequest();
        req.open("POST", "http://localhost:8000");

        var jsonData = {
            review: sentence
        };
        var jsonString = JSON.stringify(jsonData);
        console.log(jsonString)
        req.send(jsonString);

        req.onload = function() {
            if (req.status != 200) { // analyze HTTP status of the response
              alert(`Error ${req.status}: ${req.statusText}`); // e.g. 404: Not Found
            } else { // show the result
                const data = JSON.parse(req.response)
                console.log(data.sentiment_score)
            }
          };         
    }
});