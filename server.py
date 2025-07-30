# server.py
import os, uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY が未設定です")

llm = ChatOpenAI(model="gpt-3.5-turbo",
                 temperature=0.7,
                 streaming=False,
                 openai_api_key=openai_api_key)
memory = ConversationBufferMemory(return_messages=True)
chain = ConversationChain(llm=llm, memory=memory)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class ChatReq(BaseModel):
    user: str

@app.post("/chat")
async def chat(req: ChatReq):
    return {"bot": chain.predict(input=req.user)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
