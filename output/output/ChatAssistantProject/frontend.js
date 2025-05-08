// Frontend JavaScript code for Chat Assistant

const chatContainer = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

let messages = [];

sendButton.addEventListener('click', async () => {
    const userMessage = userInput.value;
    messages.push({ role: 'user', content: userMessage });
    updateChat();
    userInput.value = '';

    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages })
    });
    const data = await response.json();
    messages.push({ role: 'assistant', content: data.response });
    updateChat();
});

function updateChat() {
    chatContainer.innerHTML = '';
    messages.forEach(message => {
        const messageElement = document.createElement('div');
        messageElement.className = message.role;
        messageElement.innerText = message.content;
        chatContainer.appendChild(messageElement);
    });
}