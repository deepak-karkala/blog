# Project Planning

## 1. Idea/Vision/Purpose

**Idea:** To create an engaging GenAI application, "Chatterbox" (or a similar name), that allows users to converse with interactive, AI-powered versions of famous Sitcom, TV show, or Movie characters.

**Vision:** To provide fans and users with a fun, immersive, and authentic-feeling conversational experience, allowing them to interact with their favorite fictional personalities as if they were real. The application aims to capture the unique voice, humor, and style of each character.

**Purpose:**
*   To entertain users through novel AI-driven interactions.
*   To explore the capabilities of Large Language Models (LLMs) in persona adoption and stylistic generation.
*   To serve as a showcase for fine-tuning LLMs for specific character impersonations.
*   To create a modular and scalable platform that can be expanded with more characters and features in the future.
*   The initial character for development and launch will be **Chandler Bing** from the TV show "Friends."

---

## 2. Product Requirement Document (PRD)

**1. Introduction**
This document outlines the product requirements for "Chatterbox," a GenAI application enabling users to chat with AI representations of fictional characters.

**2. Goals**
*   Develop an MVP (Minimum Viable Product) featuring Chandler Bing.
*   Allow users to send text and image prompts.
*   Ensure AI responses are in character, contextually relevant, and engaging.
*   Provide a seamless and intuitive user interface.
*   Lay the groundwork for future expansion with more characters and features.
*   Implement in two main LLM backend stages:
    *   Stage 1: Generic LLM API (Gemini 1.5 Flash or latest equivalent).
    *   Stage 2: Custom fine-tuned Llama model.

**3. Target Users**
*   Fans of popular sitcoms, TV shows, and movies.
*   Users interested in novel AI and chatbot experiences.
*   General audience looking for lighthearted entertainment.

**4. User Stories**

*   **US1 (Character Selection):** As a user, I want to see a selection of available characters (initially Chandler Bing) so that I can choose whom I want to chat with.
*   **US2 (Visual Selection):** As a user, I want to select a character by clicking on their image/avatar so that the selection is intuitive and visual.
*   **US3 (Text Input):** As a user, I want to type a text message in an input field so that I can communicate with the selected character.
*   **US4 (Image Input):** As a user, I want to upload an image along with my text message so that the character can "see" and react to the image.
*   **US5 (Send Message):** As a user, I want to send my message (text and/or image) to the character so that I can receive a reply.
*   **US6 (Character Response):** As a user, I want to receive a response from the character that is in their specific style, personality, and humor.
*   **US7 (Streaming Response):** As a user, I want to see the character's response stream in token-by-token so that the conversation feels more dynamic and real-time.
*   **US8 (Conversation History):** As a user, I want to see the history of my conversation with the character so that I can follow the dialogue.
*   **US9 (Typing Indicator):** As a user, I want to see a "typing..." indicator when the character is generating a response so that I know the system is working.
*   **US10 (Character-Specific Greeting):** As a user, I want to be greeted by the selected character with an initial message in their style when I select them so that the experience feels immersive from the start.
*   **US11 (Character-Specific Theme):** As a user, I want the chat interface's background/theme to subtly change based on the selected character (e.g., Central Perk for Chandler) to enhance immersion.
*   **US12 (Responsive UI):** As a user, I want the application to work well on desktop, tablet, and mobile devices so that I can chat from anywhere.

**5. Functional Requirements**

*   **FR1: Character Selection:**
    *   Display available characters with images/avatars.
    *   Allow single character selection.
    *   Visually indicate the selected character.
*   **FR2: Chat Interface:**
    *   Display user messages and character responses chronologically.
    *   Differentiate user and character messages (e.g., alignment, styling, avatar).
    *   Support text messages.
    *   Support display of user-uploaded images within the chat.
*   **FR3: Message Input:**
    *   Text input field.
    *   Image upload button/functionality.
    *   Send button.
    *   Image preview before sending (with option to remove).
*   **FR4: Backend LLM Integration (Stage 1 - Generic LLM):**
    *   Connect to Gemini 1.5 Flash (or latest equivalent) API.
    *   Send user prompt (text + image context) and conversation history to the LLM.
    *   Use prompt engineering to instruct the LLM to respond in Chandler Bing's style.
*   **FR5: Backend LLM Integration (Stage 2 - Fine-tuned LLM):**
    *   Interface with a deployed custom fine-tuned Llama model (on RunPod with vLLM).
    *   Image context will be generated by Gemini 1.5 Flash and then passed to the fine-tuned Llama model.
*   **FR6: Response Streaming:**
    *   Implement server-sent events (SSE) or WebSockets to stream responses from the backend to the frontend.
*   **FR7: Conversational Memory:**
    *   Maintain conversation history/context for the duration of the session to provide coherent replies. (Managed by Langchain).
*   **FR8: Image Processing (Image Input):**
    *   Frontend: Allow image upload.
    *   Backend:
        *   For Stage 1 (Gemini): Gemini handles image input directly.
        *   For Stage 2 (Llama): Use Gemini 1.5 Flash to generate a text description of the uploaded image. This description is then fed to the fine-tuned Llama model along with the user's text.
*   **FR9: Character Theming:**
    *   Dynamically change UI background/accent colors based on the selected character.

**6. Non-Functional Requirements**

*   **NFR1 (Performance):**
    *   Response Time (P95): Character response should start streaming within 3-5 seconds of sending a message under normal load.
    *   UI should be responsive and load quickly.
*   **NFR2 (Scalability):**
    *   Backend should be designed to handle a moderate number of concurrent users (initial target: 50-100 concurrent users).
    *   Frontend deployment on Vercel should scale automatically.
    *   RunPod endpoint for Stage 2 should be configurable for scaling.
*   **NFR3 (Usability):**
    *   Intuitive and easy-to-use interface.
    *   Minimal learning curve.
*   **NFR4 (Reliability):**
    *   The application should be stable and have high uptime.
*   **NFR5 (Maintainability):**
    *   Code should be well-structured, commented, and modular to facilitate easy updates and future development (especially the backend LLM switching mechanism).
*   **NFR6 (Security):**
    *   Securely manage API keys (environment variables, secrets management).
    *   Protect against common web vulnerabilities (XSS, CSRF).
    *   Input validation to mitigate prompt injection.
    *   Secure the fine-tuned model endpoint.
*   **NFR7 (Logging & Monitoring):**
    *   Implement comprehensive logging using LangSmith for LLM interactions (prompts, responses, errors, traces).
    *   Basic application-level logging for the backend and frontend.
*   **NFR8 (Extensibility):**
    *   The system should be designed to easily add new characters in the future.

**7. Evaluation (as previously discussed)**
*   LLM-as-Judge comparing Generic LLM, SFT fine-tuned, SFT+DPO fine-tuned on humor, style adherence, relevance.

---

## 3. Detailed Description of the UI for the Application

The UI will be a single-page web application with a focus on a central chat interface. It will be built using Next.js, shadcn/ui, and Tailwind CSS, based on the Vercel AI SDK Python Streaming template.

**Key UI Components & Design Aspects:**

1.  **Overall Layout & Theme:**
    *   **Clean, Modern, Engaging:** Aesthetically pleasing and inviting.
    *   **Responsive Design:** Optimized for desktop, tablet, and mobile.
    *   **Character-Specific Theming (Subtle):**
        *   The background of the chat area will change to reflect the selected character (e.g., Central Perk couch/coffee house imagery for Chandler Bing, subtly applied so as not to distract).
        *   Accent colors in the UI might subtly shift to match the character's theme.

2.  **Header/Navigation Bar:**
    *   **Application Name/Logo:** "Chatterbox" (or chosen name) on the left.
    *   **Navigation Links (Future):** Placeholder for "Home," "Characters" (to see all), "Explore," "Settings" on the right. For MVP, these might be minimal or hidden.
    *   **User Profile/Settings Icon (Future):** Basic settings icon.

3.  **Character Selection Area:**
    *   **Location:** Positioned prominently, likely above the chat history area or as a top banner.
    *   **Display:** A horizontal row of character avatars/images.
    *   **Interaction:**
        *   Clicking a character's image selects them.
        *   **Selected State:** The selected character's image will be visually distinct (e.g., brighter border, slight zoom, checkmark overlay). Other characters may appear slightly desaturated or normal.
        *   The character's name is displayed below their image.
        *   **Default Selection:** Chandler Bing selected by default on page load for MVP.
    *   **Mobile Adaptation:** The row may become horizontally scrollable on smaller screens.

4.  **Chat Area:**
    *   **Chat History:**
        *   Main panel displaying the conversation.
        *   User messages aligned to one side (e.g., right, in a distinct color like blue).
        *   Character messages aligned to the other side (e.g., left, in a different color like light grey/off-white), prefixed with the character's avatar.
        *   Messages will support both text and inline display of user-uploaded images.
        *   Timestamps for messages (optional, subtle).
        *   Smooth scrolling and new messages appearing at the bottom.
    *   **Typing Indicator:** When the character is generating a response, a message like "Chandler is typing..." will appear, potentially with a small thematic animation (e.g., a coffee cup icon).
    *   **Initial Greeting:** Upon selecting a character, an initial greeting message from that character will appear in the chat.

5.  **Message Input Bar:**
    *   **Location:** Fixed at the bottom of the screen.
    *   **Components:**
        *   **Text Input Field:** Multi-line capable, placeholder text like "Type your message to Chandler..." (dynamically updates with character name).
        *   **Image Upload Button:** A clear icon (e.g., paperclip, image icon) next to the text field.
            *   **Image Preview:** Clicking allows file selection. Once an image is chosen, a small thumbnail preview appears above or beside the input bar, with an 'x' to remove it before sending.
        *   **Send Button:** Clearly labeled "Send" or an icon (e.g., paper airplane). Disabled if no text and no image is present.

6.  **Streaming Responses:**
    *   Character responses will appear token-by-token, creating a dynamic, real-time feel, facilitated by the Vercel AI SDK.

7.  **Loading States:**
    *   Subtle loading indicators when actions are performed (e.g., sending a message if the network is slow before the "typing" indicator appears).

8.  **Accessibility:**
    *   WCAG AA guidelines will be considered: keyboard navigability, sufficient color contrast, ARIA attributes where appropriate.

---

## 4. User/App Flow Document

**1. User Lands on Application:**
    *   **App:** Loads the main chat interface.
    *   **UI:** Displays header, character selection area (Chandler Bing selected by default), empty chat history (or with Chandler's initial greeting), and message input bar.
    *   **App:** Chandler Bing's initial greeting message ("Could I BE any more ready to chat?") appears in the chat history.

**2. User Selects a Character (Illustrative for future expansion; for MVP, Chandler is default):**
    *   **User:** Clicks on the image/avatar of a different character (e.g., "Homer Simpson").
    *   **UI:** Highlights "Homer Simpson," deselects "Chandler Bing." Character-specific theme (background, accent color) updates.
    *   **App:** Clears previous chat history (or indicates a new conversation starts).
    *   **UI:** "Homer Simpson's" initial greeting message appears. Input bar placeholder updates to "Type your message to Homer..."

**3. User Types a Text Message:**
    *   **User:** Clicks into the text input field. Types "Hey Chandler, what's up?".
    *   **UI:** Text appears in the input field. Send button becomes active.

**4. User Uploads an Image (Optional):**
    *   **User:** Clicks the "Upload Image" icon. Selects an image from their device.
    *   **UI:** Image thumbnail preview appears near the input bar with an option to remove it.
    *   **User:** (Optional) Adds text to accompany the image, e.g., "Look at this sandwich I made!"

**5. User Sends Message:**
    *   **User:** Clicks the "Send" button.
    *   **UI:**
        *   The user's message (text and/or image) appears in the chat history on the user's side.
        *   Input field clears.
        *   Image preview is removed.
        *   "Chandler is typing..." indicator appears.
    *   **App (Frontend):** Sends the message (text and image data/URL) to the backend FastAPI endpoint. Includes conversation history.

**6. Backend Processing:**
    *   **FastAPI (Backend):** Receives the request.
    *   **Langchain Orchestration:**
        *   Adds the new user message to the conversation memory.
        *   **If Image Present (Stage 1 - Gemini):** Includes image data in the prompt to Gemini.
        *   **If Image Present (Stage 2 - Llama):**
            1.  Calls Gemini 1.5 Flash with the image to get a text description.
            2.  Combines this text description with the user's text prompt.
        *   Constructs the full prompt for the selected LLM (Gemini API or Fine-tuned Llama endpoint), including system prompt (character persona), conversation history, and current user input.
    *   **LLM Interaction:**
        *   Calls the appropriate LLM.
        *   LLM generates a response.
    *   **FastAPI (Backend):** Streams the response back to the frontend using Server-Sent Events (SSE).
    *   **LangSmith:** Logs the entire interaction (request, prompt, response, any errors, latency).

**7. Frontend Receives and Displays Response:**
    *   **Next.js (Frontend - `useChat` hook):** Receives the streaming response.
    *   **UI:**
        *   "Chandler is typing..." indicator disappears (or changes to indicate response received).
        *   Character's response appears token-by-token in the chat history on the character's side, with Chandler's avatar.

**8. Conversation Continues:**
    *   User can send more messages, repeating steps 3-7. The conversation history is maintained and sent with each new request to provide context.

**Error Flows:**
*   **Network Error:** UI displays a generic error message if the backend is unreachable or there's a network issue.
*   **LLM Error:** If the LLM API returns an error, the UI displays a user-friendly message (e.g., "Chandler is a bit tongue-tied right now. Try again?"). Backend logs the detailed error.
*   **Image Upload Error:** UI displays an error if the image upload fails (e.g., file too large, unsupported format).

---

## 5. Detailed Description of Tech Stack

**1. Frontend:**
    *   **Framework:** **Next.js 14+** (App Router recommended)
        *   *Reason:* React framework for server-side rendering (SSR), static site generation (SSG), client-side rendering, API routes, performance optimizations, and excellent developer experience.
    *   **UI Components:** **shadcn/ui**
        *   *Reason:* Beautifully designed, accessible, and customizable components built on Radix UI and Tailwind CSS. Allows for rapid UI development.
    *   **Styling:** **Tailwind CSS**
        *   *Reason:* Utility-first CSS framework for rapid and consistent styling directly in markup. Works seamlessly with Next.js and shadcn/ui.
    *   **State Management:** React Context API / Zustand (or `useChat` hook's internal state for chat)
        *   *Reason:* `useChat` from Vercel AI SDK handles chat state. For other global state, React Context is built-in; Zustand for more complex needs if they arise.
    *   **AI SDK:** **Vercel AI SDK (specifically `useChat` hook)**
        *   *Reason:* Simplifies frontend implementation of streaming chat completions, managing chat state, and user input.
    *   **Deployment:** **Vercel**
        *   *Reason:* Seamless deployment for Next.js applications, CI/CD, global CDN, serverless functions.

**2. Backend:**
    *   **Framework:** **Python FastAPI**
        *   *Reason:* Modern, fast (high-performance) web framework for building APIs with Python, based on Pydantic and Starlette. Asynchronous support is ideal for I/O-bound LLM calls. Automatic data validation and OpenAPI docs.
    *   **LLM Orchestration:** **Langchain (Python)**
        *   *Reason:* Framework for developing applications powered by language models. Handles:
            *   Conversational memory (e.g., `ConversationBufferMemory`).
            *   Prompt templating and management.
            *   Chains for structuring LLM calls.
            *   Integrations with LLM providers and custom LLM endpoints.
    *   **Stage 1 LLM:** **Google Gemini 1.5 Flash API (or latest Flash variant)**
        *   *Reason:* Multimodal (handles text and image), fast, cost-effective, large context window. Accessed via its Python SDK.
    *   **Stage 2 LLM (Base Model for Fine-tuning):** **Llama 3 (e.g., 8B variant)**
        *   *Reason:* Strong open-source base model with good performance for fine-tuning.
    *   **Stage 2 LLM (Fine-tuning & Deployment):** **RunPod**
        *   *Reason:* Cost-effective GPU compute for fine-tuning Llama models. Provides serverless GPU endpoints for deploying the fine-tuned model.
    *   **Stage 2 LLM (Inference Optimization):** **vLLM**
        *   *Reason:* High-throughput and memory-efficient serving engine for LLMs. Will be used to serve the fine-tuned Llama model on RunPod.
    *   **Image Processing (for Stage 2 Llama):** **Google Gemini 1.5 Flash API**
        *   *Reason:* To generate text descriptions from user-uploaded images, which are then fed to the text-only fine-tuned Llama model.
    *   **WSGI/ASGI Server:** **Uvicorn** (comes with FastAPI)
        *   *Reason:* ASGI server for running FastAPI applications.

**3. Logging & Monitoring:**
    *   **LangSmith**
        *   *Reason:* Specifically designed for LLM applications, provides tracing, debugging, monitoring, and data collection for Langchain-based apps.

**4. Security Considerations (to be implemented):**
    *   **API Key Management:** Use environment variables and Vercel/RunPod secret management.
    *   **Input Sanitization:** Langchain prompt templates help, but further validation where user input is directly used.
    *   **CORS:** Configure appropriately in FastAPI.
    *   **HTTPS:** Handled by Vercel and RunPod for deployed endpoints.
    *   **Endpoint Protection (RunPod):** Use API keys or other authentication mechanisms for the custom model endpoint.

---

## 6. Recommended Directory Structure

This structure aims for clarity and separation of concerns for both the frontend (Next.js) and backend (Python FastAPI) parts of the application.

```
chatterbox-app/
├── frontend/                     # Next.js Application
│   ├── app/                      # Next.js App Router
│   │   ├── (chat)/               # Route group for chat interface
│   │   │   ├── [characterId]/    # Dynamic route for character (optional for MVP if only Chandler)
│   │   │   │   └── page.tsx      # Main chat page component
│   │   │   └── layout.tsx        # Layout for the chat section
│   │   ├── api/                  # Next.js API Routes (if any specific to frontend, e.g. proxying)
│   │   │   └── chat/             # If using Next.js API routes instead of separate FastAPI for Vercel deployment
│   │   │       └── route.ts      # API route for chat streaming (alternative to separate FastAPI)
│   │   ├── layout.tsx            # Root layout
│   │   └── page.tsx              # Root page (e.g., landing page or redirect to chat)
│   ├── components/               # Reusable React components
│   │   ├── ui/                   # Components from shadcn/ui (will be auto-generated here)
│   │   ├── chat/                 # Chat-specific components
│   │   │   ├── CharacterSelector.tsx
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── MessageInput.tsx
│   │   │   └── TypingIndicator.tsx
│   │   └── common/               # Common components like Header, Footer
│   ├── contexts/                 # React Context providers (if needed beyond useChat)
│   ├── hooks/                    # Custom React hooks
│   ├── lib/                      # Utility functions, helper scripts (e.g., Vercel AI SDK config)
│   ├── public/                   # Static assets (images, fonts, character avatars, backgrounds)
│   │   └── characters/
│   │       └── chandler/
│   │           ├── avatar.png
│   │           └── background.jpg
│   ├── styles/                   # Global styles, Tailwind CSS config
│   │   └── globals.css
│   ├── next.config.mjs           # Next.js configuration
│   ├── postcss.config.js         # PostCSS configuration (for Tailwind)
│   ├── tailwind.config.ts        # Tailwind CSS configuration
│   └── tsconfig.json             # TypeScript configuration
│
├── backend/                      # Python FastAPI Application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app instance, middleware, root routers
│   │   ├── api/                  # API routers/endpoints
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── endpoints/
│   │   │           ├── __init__.py
│   │   │           └── chat.py     # Chat endpoint logic
│   │   ├── core/                 # Core logic, configuration
│   │   │   ├── __init__.py
│   │   │   └── config.py         # Application configuration (API keys, settings)
│   │   ├── schemas/              # Pydantic schemas for request/response validation
│   │   │   ├── __init__.py
│   │   │   └── chat_schemas.py
│   │   ├── services/             # Business logic services
│   │   │   ├── __init__.py
│   │   │   └── llm_service.py    # LLM interaction logic (abstraction layer, Gemini, Llama calls)
│   │   ├── prompts/              # Character-specific prompt templates/texts
│   │   │   ├── __init__.py
│   │   │   └── chandler_bing.txt # Or .py file defining prompt constants
│   │   └── utils/                # Utility functions for the backend
│   │       └── __init__.py
│   ├── tests/                    # Backend tests
│   ├── .env                      # Environment variables (DO NOT COMMIT if sensitive)
│   ├── .env.example              # Example environment variables
│   ├── requirements.txt          # Python dependencies
│   └── Dockerfile                # Optional: For containerizing the backend
│
└── vercel.json                   # Vercel deployment configuration (if needed for advanced setups)
```

**Explanation of Key Directories:**

**`frontend/` (Next.js)**
*   **`app/`**: Uses Next.js App Router.
    *   `(chat)/[characterId]/page.tsx`: The main page where the chat happens. `[characterId]` makes it dynamic if you add more characters. For MVP with only Chandler, it could be `app/(chat)/chandler/page.tsx` or simplified.
    *   `api/chat/route.ts`: This is an **alternative** if you choose to host the Python backend logic within Next.js API routes using Vercel's Python serverless functions. If you have a separate FastAPI deployment (e.g., on another service, or even a separate Vercel deployment for Python), this might not be used, and the `useChat` hook would point to that external FastAPI URL. Given your plan for Stage 2 with RunPod, a separate FastAPI backend is more likely, but this shows the Next.js native option.
*   **`components/`**: For all React components.
    *   `ui/`: Where `shadcn/ui` will place its components.
    *   `chat/`: Custom components specifically for the chat interface.
*   **`public/characters/[characterName]/`**: Good place to store character-specific static assets like avatars and background images.
*   **`lib/`**: For utility functions, including any client-side configuration for the Vercel AI SDK.

**`backend/` (Python FastAPI)**
*   **`app/`**: Main application module.
    *   **`main.py`**: Entry point for the FastAPI application.
    *   **`api/v1/endpoints/chat.py`**: Defines the `/chat` API endpoint. Organized by version for future API evolution.
    *   **`core/config.py`**: For managing settings and environment variables (like API keys).
    *   **`schemas/`**: Pydantic models for data validation.
    *   **`services/llm_service.py`**: This is crucial. It will contain the `CharacterLLMInterface` and its implementations (`GenericAPICharacterLLM`, `FineTunedCharacterLLM`) for interacting with Gemini and the fine-tuned Llama model. This is where Langchain orchestration will primarily live.
    *   **`prompts/`**: To store the actual text for system prompts for each character.
*   **`requirements.txt`**: Python package dependencies.
*   **`.env`**: For storing environment variables locally (API keys, etc.). Ensure this is in `.gitignore`.
*   **`Dockerfile`**: Would be relevant if you decide to containerize the FastAPI backend for deployment on services other than Vercel's serverless functions (e.g., if the backend grows too complex for serverless or for consistency with RunPod deployment style).

**Root Directory:**
*   `vercel.json`: For Vercel-specific deployment configurations, like configuring rewrites or specifying Python version if deploying FastAPI to Vercel.

**Considerations:**

*   **Monorepo vs. Separate Repos:** This structure assumes a monorepo (frontend and backend in the same repository). You could also have them as separate repositories if preferred, especially if different teams work on them. For a solo developer or small team, a monorepo is often easier to manage initially.
*   **Next.js API Routes for Backend:** If you deploy the FastAPI backend directly on Vercel using its Python Serverless Functions, your FastAPI app might be structured to be callable from Next.js API routes (e.g., the `frontend/app/api/chat/route.ts` would import and run the FastAPI app or specific parts of it). The current structure is more aligned with a potentially separate FastAPI deployment, which gives more flexibility for complex backends or alternative hosting. Your FastAPI app under `backend/` can be deployed to Vercel as well.
*   **RunPod Deployment:** The fine-tuned model deployed on RunPod will be a separate entity. The `backend/services/llm_service.py` will make HTTP requests to the RunPod endpoint.

This structure should provide a good starting point for your coding agent. Let me know if you have any questions or want to adjust any part of it!


## 7. Detailed Phase-by-Phase Implementation Plan

**Phase 0: Setup & Foundation**
*   **Step 0.1: Project Initialization:**
    *   Set up Git repository.
    *   Initialize Next.js project (`create-next-app`).
    *   Initialize FastAPI project structure.
*   **Step 0.2: Install Core Dependencies:**
    *   Frontend: `ai` (Vercel AI SDK), `tailwindcss`, `shadcn-ui` setup.
    *   Backend: `fastapi`, `uvicorn`, `langchain`, `google-generativeai`, `python-dotenv`.
*   **Step 0.3: Vercel & RunPod Account Setup:**
    *   Ensure accounts are ready and billing is set up if necessary.
*   **Step 0.4: LangSmith Setup:**
    *   Create a LangSmith account and project. Configure API keys for backend integration.
*   **Step 0.5: API Key Management:**
    *   Set up `.env` files for local development (Gemini API key, LangSmith keys).
    *   Plan for Vercel and RunPod environment variable configuration.
*   **Verification:** Basic Next.js app runs. Basic FastAPI app runs. LangSmith dashboard is accessible.

**Phase 1: Frontend UI Shell & Basic Chat (No LLM yet)**
*   **Step 1.1: Implement Basic Layout:**
    *   Create header, character selection placeholder, chat area, and message input bar using Next.js components and Tailwind CSS.
    *   Reference the Vercel AI SDK Python Streaming template structure.
*   **Step 1.2: Character Selection UI (Static):**
    *   Implement the visual row for character selection (initially just Chandler Bing image/avatar).
    *   Implement click functionality to visually mark Chandler as selected.
*   **Step 1.3: Message Display UI:**
    *   Create components for user message bubbles and character message bubbles.
    *   Allow manual adding of messages to a local state to test display.
*   **Step 1.4: Message Input UI:**
    *   Implement text input field, image upload button (no functionality yet), and send button.
    *   Basic state management for the input field.
*   **Step 1.5: Integrate `useChat` Hook (Basic):**
    *   Integrate the `useChat` hook from Vercel AI SDK, pointing to a dummy FastAPI endpoint that returns a hardcoded streaming response.
    *   Ensure messages are added to the chat history via the hook.
*   **Verification:** UI shell is in place. Character selection visually works. User can type, "send" (locally), and see messages appear. `useChat` streams hardcoded response from a dummy backend endpoint.

**Phase 2: Backend Stage 1 (Gemini LLM) & Frontend Integration**
*   **Step 2.1: FastAPI Endpoint for Chat:**
    *   Create a FastAPI streaming endpoint (`/api/chat`) that accepts user messages and conversation history.
*   **Step 2.2: Langchain Setup for Gemini:**
    *   Initialize Langchain with Gemini 1.5 Flash.
    *   Create prompt templates for Chandler Bing.
    *   Implement `ConversationBufferMemory` for context.
    *   Create a Langchain chain to process input, manage memory, and call Gemini.
*   **Step 2.3: Implement Streaming Response from Gemini:**
    *   Modify the FastAPI endpoint to stream responses from the Langchain/Gemini setup using Server-Sent Events (SSE).
*   **Step 2.4: Connect Frontend to Backend:**
    *   Update the `useChat` hook in Next.js to point to the actual FastAPI `/api/chat` endpoint.
*   **Step 2.5: Implement Image Handling (Stage 1 - Gemini):**
    *   Frontend: Allow actual image selection and send image data (e.g., base64 string or multipart/form-data) to the backend.
    *   Backend: Modify FastAPI endpoint and Langchain chain to accept image data and pass it to Gemini 1.5 Flash along with text.
    *   Frontend: Display uploaded images in user messages.
*   **Step 2.6: LangSmith Integration:**
    *   Ensure all Langchain interactions are being traced and logged in LangSmith.
*   **Step 2.7: Character-Specific Greeting & Theming (Chandler):**
    *   Implement initial greeting from Chandler on load.
    *   Implement background theme change for Chandler.
*   **Verification:** Full end-to-end text and image chat with Chandler Bing using Gemini 1.5 Flash. Responses are in character. Streaming works. Conversation history is maintained. LangSmith shows traces. UI theming for Chandler is active.

**Phase 3: Backend Stage 2 (Fine-tuned Llama Model)**
*   **Step 3.1: Fine-tune Llama Model:**
    *   Prepare fine-tuning dataset for Chandler Bing (already available).
    *   Use RunPod (or local GPU if powerful enough) to fine-tune a Llama 3 (e.g., 8B) model using SFT.
    *   (Optional - Advanced): Experiment with DPO if preference data is available/creatable.
*   **Step 3.2: Deploy Fine-tuned Model on RunPod with vLLM:**
    *   Containerize the fine-tuned model with a vLLM server.
    *   Deploy this container as a serverless endpoint on RunPod.
    *   Secure the endpoint.
*   **Step 3.3: Create Backend Abstraction for LLM Switching:**
    *   Refactor the FastAPI backend to include an abstraction layer for LLM services (as discussed, e.g., `CharacterLLMInterface` with `GenericAPICharacterLLM` and `FineTunedCharacterLLM` implementations).
    *   Use environment variables to switch between Gemini and the fine-tuned model.
*   **Step 3.4: Implement `FineTunedCharacterLLM` Service:**
    *   This service will call the deployed RunPod endpoint.
    *   Langchain will still manage conversation memory and orchestrate calls.
*   **Step 3.5: Implement Image Handling (Stage 2 - Llama):**
    *   In the `FineTunedCharacterLLM` service flow:
        1.  Receive image data from the frontend.
        2.  Call Gemini 1.5 Flash API to get a text description of the image.
        3.  Combine this text description with the user's text prompt.
        4.  Send the combined text to the fine-tuned Llama model on RunPod.
*   **Verification:** Application can now use the fine-tuned Llama model for Chandler. Image inputs are processed via Gemini description before Llama. Responses are high quality and deeply in character. Easy to switch back to Gemini via config. LangSmith continues to log.

**Phase 4: Polishing, Testing & Deployment**
*   **Step 4.1: Thorough Testing:**
    *   Conduct comprehensive testing across browsers and devices.
    *   Functional testing (all user stories).
    *   Performance testing (response times).
    *   Usability testing.
    *   LLM Evaluation (as per plan: LLM-as-Judge).
*   **Step 4.2: UI/UX Refinements:**
    *   Implement smooth transitions and animations.
    *   Address any UI inconsistencies or bugs.
    *   Refine character themes and greetings.
*   **Step 4.3: Security Hardening:**
    *   Review all security considerations (NFR6).
    *   Perform basic vulnerability checks.
*   **Step 4.4: Finalize Logging & Monitoring:**
    *   Ensure LangSmith setup is robust.
    *   Add any necessary application-level logging.
*   **Step 4.5: Prepare for Deployment:**
    *   Configure Vercel environment variables for production (API keys, backend URL).
    *   Configure RunPod endpoint for production use (scaling, security).
*   **Step 4.6: Deploy to Production:**
    *   Deploy Next.js frontend to Vercel.
    *   Ensure FastAPI backend (if separate from Vercel functions for LLM calls, or if Vercel hosts the part calling RunPod) is deployed and accessible.
    *   Ensure RunPod endpoint is live and stable.
*   **Verification:** Application is live and functioning correctly in a production environment. All features work as expected.

**Phase 5: Post-Launch & Future Iterations (Beyond initial scope for coding agent but good to note)**
*   **Step 5.1: Monitor Performance & User Feedback.**
*   **Step 5.2: Iterate on Fine-tuned Model (if needed).**
*   **Step 5.3: Plan for Adding New Characters.**
*   **Step 5.4: Explore Advanced Features (e.g., voice input/output, more complex memory).**

---

