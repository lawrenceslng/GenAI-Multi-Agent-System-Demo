from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Travel Planning Chat Assistant! Ask me about destinations, activities, or packing tips."

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')

    # Processing user message and generating a response
    response = ""
    if "destination" in user_message.lower():
        response = "I recommend visiting Kyoto, Japan for a cultural experience!"
    elif "activities" in user_message.lower():
        response = "You can try hiking, city tours, or relaxing on a beach!"
    elif "packing tips" in user_message.lower():
        response = "Pack light! Include travel-sized toiletries, comfortable shoes, and weather-appropriate clothing."
    else:
        response = "I'm not sure about that. Please ask me about destinations, activities, or packing tips!"

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)