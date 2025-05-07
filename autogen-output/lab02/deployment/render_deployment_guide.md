# Render Deployment Guide for Chat Assistant

## Prerequisites
1. A Render.com account. If you don’t already have one, sign up at [Render](https://render.com/).
2. The Chat Assistant codebase ready for deployment.
3. Access to the OpenAI API key to enable the chatbot's functionality.

## Deployment Steps

1. **Log in to Render:**
   Log in to your Render.com account.

2. **Create a New Web Service:**
   - Navigate to the dashboard.
   - Click on **New** > **Web Service**.

3. **Connect GitHub Repository:**
   - Choose the `lab02` GitHub repository containing the Chat Assistant code.
   - Select the `main` branch or the branch where the chatbot application resides.

4. **Set Up Deployment Settings:**
   - Name your service (e.g., `chat-assistant`).
   - Select a runtime environment.
     - Environment: Python 3.8+.
   - Build Command: 
     ```bash
     pip install -r requirements.txt
     ```
   - Start Command: 
     ```bash
     python main.py
     ```

5. **Add Environment Variables:**
   Navigate to the environment settings of the service and add the following:
   - `OPENAI_API_KEY`: Your OpenAI API key.

6. **Deploy the Service:**
   - Click **Create Web Service**.
   - Wait for the deployment to complete and ensure it starts without error.

7. **Test Your Chat Assistant:**
   - Visit the live URL provided by Render.com.
   - Test endpoint: `/chat` (POST request with JSON payload `{"message": "Your question"}`).

## Post-Deployment
1. Add the Render.com live link to the README under the "Deployment" section.
2. Monitor application logs for debugging if issues arise during testing.

---

For any further deployment guidance, refer to the Render documentation or contact the team lead.