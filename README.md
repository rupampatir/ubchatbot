# Rasa Chatbot for UB CSE Department

This repository contains the code for a Rasa-based chatbot designed for the Computer Science and Engineering Department at the University of Buffalo. The chatbot is capable of answering queries related to courses, faculty, admissions, and general department information.

## Installation

Before running the chatbot, ensure you have all the necessary dependencies installed.

pip install -r requirements.txt

## Running the Chatbot

To run the chatbot, you need to start the Rasa server and the action server. Follow these steps:

1. Initailise rasa (only once)
    rasa init

2. Start the Rasa server:
rasa run -m models --enable-api --cors "*" --debug

3. In a new terminal window, start the Rasa action server:
rasa run actions --cors "*" --debug

4. After both servers are up and running open the `index.html` file located in the `ChatWidget` folder on your browser to interact with the chatbot.
