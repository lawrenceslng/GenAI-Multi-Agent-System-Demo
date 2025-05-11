# Import necessary libraries
from flask import Flask, request, jsonify
import openai

# Initialize the Flask application
app = Flask(__name__)

# Set up the OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Define the route for the Chat Assistant
@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint to handle chat requests"""
    user_input = request.json.get('message', '')
    
    if not user_input:
        return jsonify({"error": "Message field is required."}), 400

    try:
        # Call OpenAI API for response
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"You are a helpful assistant specialized in technology topics. {user_input}",
            max_tokens=150
        )

        # Extract response text
        assistant_reply = response.choices[0].text.strip()

        return jsonify({"response": assistant_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)