from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins
    allow_credentials=True,
    allow_methods=['*'],  # Allows all methods
    allow_headers=['*'],  # Allows all headers
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return "<h1>Chat Assistant API</h1>"

@app.post("/send")
async def send_message(request: Request):
    data = await request.json()
    user_message = data.get('message')
    # Here you would integrate with LiteLLM to get a response
    assistant_response = f'You said: {user_message}'  # Placeholder response
    return assistant_response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)