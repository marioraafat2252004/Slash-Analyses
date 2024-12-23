from fastapi import APIRouter, HTTPException
from controllers.chatController import handle_chat
from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: str
    message: str

router = APIRouter()

@router.post("/message", summary="Send a message to the chatbot")
async def send_message(chat_request: ChatRequest):
    try:
        response = handle_chat(chat_request.user_id, chat_request.message)
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
