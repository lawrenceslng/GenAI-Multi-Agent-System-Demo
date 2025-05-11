// Backend logic for the health and fitness Chat Assistant

const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

// Sample responses for health inquiries
const responses = {
    'what is a balanced diet?': 'A balanced diet includes a variety of foods in the right proportions to provide the nutrients you need. This typically includes fruits, vegetables, whole grains, and lean proteins.',
    'how much water should I drink?': 'It is generally recommended to drink at least 8 glasses (2 liters) of water per day for proper hydration.',
};

app.post('/ask', (req, res) => {
    const question = req.body.question.toLowerCase();
    const answer = responses[question] || 'I am sorry, I do not have an answer for that.';
    res.send({ answer });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});