# Backend Code Explanation

This document provides a detailed explanation of the backend code for the application. The backend is built with FastAPI.

## `api/app/main.py`

This file is the main entry point for the FastAPI application. It sets up the application, including logging, CORS, and API routing.

### Code

```python
# backend/app/main.py
import sys
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# This must happen before the other app imports to set up the path
api_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if api_dir not in sys.path:
    sys.path.insert(0, api_dir)

from app.api.v1.endpoints.chat import router as api_router
from app.core.config import settings
from app.core.logging_config import setup_logging

# Load environment variables from .env file as early as possible
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# ... (LangSmith and OpenAI API key verification logs)

app = FastAPI(
    title=settings.APP_NAME,
    openapi_url="/api/v1/openapi.json"
)
logger.info("FastAPI application starting up...")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, you should restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    logger.info("Root endpoint '/' was called.")
    return {
        "message": f"Welcome to {settings.APP_NAME}. "
                   "Visit /docs for API documentation."
    }

logger.info("FastAPI application configured and ready.")

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup...")
    pass

## `api/app/api/v1/endpoints/chat.py`

This file defines the main `/chat` endpoint for the application. It handles incoming chat requests, applies input guardrails, and streams responses back from the LLM service.

### Code

```python
# backend/app/api/v1/endpoints/chat.py
import logging
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.schemas.chat_schemas import ChatRequest
from app.services.llm_service import LLMService
from app.core.guardrails_config import (
    INPUT_DENYLIST_KEYWORDS, CANNED_RESPONSE_INPUT_TRIGGERED
)

logger = logging.getLogger(__name__)
router = APIRouter()

DEFAULT_SESSION_ID = "default_frontend_session"

def get_llm_service():
    return LLMService()

async def create_canned_stream(response_text: str):
    """Helper to stream a canned response in the SDK-expected format."""
    json_stringified_token = json.dumps(response_text)
    formatted_chunk = f"0:{json_stringified_token}\n"
    yield formatted_chunk

@router.post("/chat")
async def handle_chat_streaming(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service)
):
    # ... session and character ID handling ...

    if not request.messages:
        # ... handle empty message list ...
        canned_response = (
            "Could I BE any more confused? You didn't say anything!"
        )
        return StreamingResponse(
            create_canned_stream(canned_response),
            media_type="text/plain"
        )

    current_user_input = request.messages[-1].content

    # Input Guardrail Check
    lower_user_input = current_user_input.lower()
    for keyword in INPUT_DENYLIST_KEYWORDS:
        if keyword.lower() in lower_user_input:
            logger.warning("Input Guardrail triggered...")
            return StreamingResponse(
                create_canned_stream(CANNED_RESPONSE_INPUT_TRIGGERED),
                media_type="text/plain"
            )

    raw_token_generator = llm_service.async_generate_streaming_response(
        user_input=current_user_input,
        image_notes=request.image_context_notes,
        conversation_id=session_id_to_use,
        character_id=character_id_to_use
    )

    async def sdk_formatted_stream_generator():
        async for token in raw_token_generator:
            if token is not None:
                json_stringified_token = json.dumps(token)
                formatted_chunk = f"0:{json_stringified_token}\n"
                yield formatted_chunk

    return StreamingResponse(
        sdk_formatted_stream_generator(), media_type="text/plain"
    )
```

### Explanation

-   **Router and Dependencies**:
    -   An `APIRouter` is created to define the endpoint.
    -   A `get_llm_service` dependency is defined to provide an instance of the `LLMService`.
-   **`create_canned_stream` Helper**:
    -   This asynchronous helper function takes a string and yields it in the format expected by the Vercel AI SDK frontend (`0:"<json_string>"\n`). This is used for sending predefined responses, like guardrail triggers or error messages.
-   **`/chat` Endpoint**:
    -   This is a POST endpoint that receives a `ChatRequest` object.
    -   **Session and Character Handling**: It determines the session ID and character ID to use, with default values if they are not provided.
    -   **Input Validation**: It checks if the `messages` list is empty and, if so, returns a canned response.
    -   **Input Guardrails**: It checks the user's latest message against a denylist of keywords (`INPUT_DENYLIST_KEYWORDS`). If a keyword is found, it logs a warning and returns a predefined canned response.
    -   **LLM Service Call**: It calls the `async_generate_streaming_response` method of the `LLMService` to get a raw token generator from the language model.
    -   **Streaming Response**:
        -   The `sdk_formatted_stream_generator` async generator function iterates through the raw tokens from the LLM service.
        -   It formats each token into the AI SDK's streaming format and yields it.
        -   A `StreamingResponse` is returned, which sends the formatted tokens to the client as they are generated.

## `api/app/core/config.py`

This file defines the application's configuration using Pydantic's `BaseSettings`. It allows for managing settings from environment variables and `.env` files.

### Code

```python
# backend/app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "Chatterbox Backend"
    
    # OpenAI / vLLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o-mini"

    # LangSmith Configuration
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: Optional[str] = "Chatterbox"
    LANGCHAIN_VERBOSE: bool = False

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings()
```

### Explanation

-   **`Settings` Class**:
    -   Inherits from `pydantic_settings.BaseSettings`, which automatically reads settings from environment variables.
    -   Defines various configuration variables with type hints and default values.
    -   **OpenAI/vLLM**: `OPENAI_API_KEY`, `OPENAI_BASE_URL` (for custom endpoints), and `OPENAI_MODEL_NAME`.
    -   **LangSmith**: `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_ENDPOINT`, `LANGCHAIN_PROJECT`, and `LANGCHAIN_VERBOSE` for observability.
-   **`model_config`**:
    -   This is a Pydantic V2 feature that configures the behavior of the `Settings` class.
    -   `env_file=".env"`: Specifies that settings should be loaded from a `.env` file.
    -   `extra='ignore'`: Ignores any extra environment variables that are not defined in the `Settings` class.
-   **`settings` Instance**:
    -   A global instance of the `Settings` class is created, which can be imported and used throughout the application to access configuration values.

## `api/app/core/guardrails_config.py`

This file centralizes the configuration for content safety and guardrails. It defines denylists for both input and output, as well as the canned responses to be used when a guardrail is triggered.

### Code

```python
# backend/app/core/guardrails_config.py

# --- Input Deny-List ---
# Keywords/phrases that, if found in user input, will trigger a canned response
# and prevent the input from going to the LLM.
INPUT_DENYLIST_KEYWORDS = [
    "kill yourself",
    "i want to die",
    # ... more keywords
]

# --- Output Deny-List ---
# Keywords/phrases that, if found in LLM output, will trigger the output
# to be replaced or modified.
OUTPUT_DENYLIST_KEYWORDS = [
    "i am an ai language model",
    "i cannot have opinions",
    "as a large language model",
    # ... more keywords
]

# --- Canned Responses (In Character for Chandler) ---

CANNED_RESPONSE_INPUT_TRIGGERED = (
    "Whoa there, pal! Could that topic BE any more out of left field? "
    # ... more text
)

CANNED_RESPONSE_OUTPUT_TRIGGERED = (
    "Yikes! Did I just say that out loud? My brain-to-mouth filter must be "
    # ... more text
)
```

### Explanation

-   **`INPUT_DENYLIST_KEYWORDS`**:
    -   A list of strings that should not be present in the user's input.
    -   If any of these keywords are detected (case-insensitively), the application will not send the request to the LLM and will instead return a predefined canned response.
    -   This is a basic form of input moderation to filter out harmful or inappropriate content.
-   **`OUTPUT_DENYLIST_KEYWORDS`**:
    -   A list of strings that should not be present in the LLM's output.
    -   This is used to prevent the model from breaking character (e.g., saying "I am an AI language model") or generating undesirable content.
    -   If these keywords are detected in the LLM's response, the response is replaced with a canned response.
-   **Canned Responses**:
    -   `CANNED_RESPONSE_INPUT_TRIGGERED`: The message sent to the user when their input triggers the input denylist.
    -   `CANNED_RESPONSE_OUTPUT_TRIGGERED`: The message sent to the user if the LLM's output triggers the output denylist.
    -   These responses are written in the persona of the selected character (in this case, Chandler Bing) to maintain the user experience.

## `api/app/core/logging_config.py`

This file sets up a basic, standardized logging configuration for the entire application.

### Code

```python
# backend/app/core/logging_config.py
import logging
import sys

def setup_logging(log_level=logging.INFO):
    """Configures basic logging for the application."""

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
```

### Explanation

-   **`setup_logging` Function**:
    -   This function configures the root logger for the application.
    -   It takes an optional `log_level` argument, which defaults to `logging.INFO`.
    -   **Log Format**: It defines a standard format for log messages, including the timestamp, logger name, log level, and the message itself.
    -   **Handler Management**: It clears any existing handlers from the root logger to prevent duplicate log output, which can happen if the function is called multiple times.
    -   **`basicConfig`**: It uses `logging.basicConfig` to set up the configuration.
        -   `level`: Sets the minimum level of messages to be logged.
        -   `format`: Applies the defined log format.
        -   `handlers`: Specifies that logs should be sent to standard output (`sys.stdout`) via a `StreamHandler`.

## `api/app/services/llm_service.py`

This is the core service of the application, responsible for all interactions with the language model. It handles session history, system prompts, guardrails, and both streaming and non-streaming responses.

### Code

```python
# backend/app/services/llm_service.py
import os
import logging
import asyncio
# ... other imports
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

def load_system_prompt(character_id: str = "chandler"):
    # ... implementation ...

module_level_session_histories: Dict[str, InMemoryChatMessageHistory] = {}

def check_output_for_violations(text_chunk: str) -> bool:
    # ... implementation ...

class LLMService:
    def __init__(self):
        # ... initialization of OpenAI client and LangSmith wrapper ...

    @traceable()
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        # ... implementation ...

    @traceable()
    def _prepare_input_with_image_context(self, user_input: str, image_notes: Optional[str]) -> str:
        # ... implementation ...

    @traceable()
    def _build_openai_messages(self, history: InMemoryChatMessageHistory, user_input_combined: str, system_prompt: str) -> List[Dict[str, Any]]:
        # ... implementation ...

    @traceable(run_type="llm")
    async def generate_response(
        self, user_input: str, image_notes: Optional[str] = None,
        conversation_id: str = "default_conv",
        character_id: str = "chandler"
    ):
        # ... non-streaming response generation ...

    @traceable(run_type="llm")
    async def async_generate_streaming_response(
        self, user_input: str, image_notes: Optional[str] = None,
        conversation_id: str = "default_conv",
        character_id: str = "chandler"
    ) -> AsyncGenerator[str, None]:
        # ... streaming response generation ...
```

### Explanation

-   **`load_system_prompt`**: This function reads a system prompt from a text file in the `api/app/prompts` directory based on the `character_id`. This allows for different personas and instructions for each character.
-   **`module_level_session_histories`**: A dictionary that acts as a simple in-memory store for chat histories, with each session ID mapping to an `InMemoryChatMessageHistory` object from LangChain.
-   **`check_output_for_violations`**: This function checks a chunk of text from the LLM's output against the `OUTPUT_DENYLIST_KEYWORDS` to enforce output guardrails.
-   **`LLMService` Class**:
    -   **`__init__`**: The constructor initializes the OpenAI client. It uses the `OPENAI_API_KEY` and optional `OPENAI_BASE_URL` from the settings. If a `LANGCHAIN_API_KEY` is present, it wraps the OpenAI client with LangSmith's `wrap_openai` for tracing and observability.
    -   **`get_session_history`**: Retrieves the chat history for a given `session_id` from the in-memory store, creating a new one if it doesn't exist.
    -   **`_prepare_input_with_image_context`**: If `image_notes` are provided (from image analysis on the frontend), this method appends them to the user's text input.
    -   **`_build_openai_messages`**: This method constructs the list of messages to be sent to the OpenAI API. It starts with the system prompt, adds the existing chat history, and finally appends the latest user input.
    -   **`generate_response`**: An `async` method for generating a non-streaming response. It gets the history, builds the messages, calls the OpenAI API, checks the response for guardrail violations, and updates the history.
    -   **`async_generate_streaming_response`**: The primary method used by the chat endpoint.
        -   It prepares the prompts and history similar to the non-streaming method.
        -   It calls the OpenAI client with `stream=True`.
        -   It accumulates the streamed tokens into a full response. After the stream is complete, it checks the full response for guardrail violations.
        -   If a violation is found, it yields a canned response and updates the history accordingly.
        -   If no violation is found, it yields the original tokens one by one and then updates the history with the full, valid response.
        -   All public methods are decorated with `@traceable` from LangSmith, which automatically creates traces for the function calls. 
```

## `api/app/schemas/chat_schemas.py`

This file defines the Pydantic models used for data validation and serialization in the API endpoints.

### Code

```python
from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    image_context_notes: Optional[str] = None
    session_id: Optional[str] = None
    character_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
```

### Explanation

-   **`Message`**:
    -   A simple model representing a single chat message.
    -   `role`: The role of the message sender (e.g., "user" or "assistant").
    -   `content`: The text content of the message.
-   **`ChatRequest`**:
    -   This model represents the structure of the JSON body expected by the `/chat` endpoint.
    -   `messages`: A list of `Message` objects, representing the conversation history.
    -   `image_context_notes`: An optional string containing notes from image analysis.
    -   `session_id`: An optional string to identify the user's session.
    -   `character_id`: An optional string to identify the selected character.
-   **`ChatResponse`**:
    -   A model that could be used for a non-streaming response, containing the `reply` from the assistant. This is not used by the primary streaming endpoint but is kept for potential future use.

## `api/app/prompts/`

This directory contains the system prompts that define the personality and behavior of each character the user can interact with. Each text file corresponds to a `character_id`.

### Purpose

System prompts are a crucial part of controlling the output of a language model. They provide the initial instructions that guide the model's responses throughout a conversation. In this application, they are used to make the LLM adopt a specific persona.

### Example: `chandler.txt`

```text
You are Chandler Bing from the TV show Friends. Your personality is sarcastic, witty, and often self-deprecating. You make jokes frequently, sometimes at inappropriate times. You are known for your catchphrase "Could I BE any more [adjective]?". You are currently chatting with a user who is a fan.

Keep your responses in character. Be funny, use sarcasm, and try to mimic Chandler's speaking style and common phrases. If you are unsure how to respond, a bit of awkward humor is fine. Do not break character. Do not say you are an AI.

Remember key details about your life: You work in "statistical analysis and data reconfiguration" (though you find it boring). Your best friends are Joey, Ross, Monica (Ross's sister, your eventual wife), Rachel, and Phoebe. You lived with Joey for a long time. You have a complicated relationship with your parents.

**It is crucial that you consider the *entire* preceding conversation history to ensure your responses are relevant, contextually appropriate, and avoid repetition. Refer to earlier messages when it makes sense to do so to create a continuous and engaging conversation.**

Engage with the user in a way that is typical of Chandler.
```

### Explanation of the Prompt Structure

-   **Persona Definition**: The prompt starts by clearly stating who the model should be ("You are Chandler Bing..."). It describes the key personality traits (sarcastic, witty).
-   **Behavioral Rules**: It gives explicit instructions on how to behave ("Keep your responses in character," "Do not break character," "Do not say you are an AI").
-   **Knowledge Base**: It provides key details about the character's life and relationships to help the model maintain consistency.
-   **Contextual Awareness**: It explicitly instructs the model to consider the entire conversation history to provide relevant and engaging responses. This is a critical instruction for creating a believable chat experience.
-   **Engagement Style**: It concludes by reinforcing the desired style of interaction.