# Frontend Code Explained

This document provides a comprehensive explanation of the frontend codebase for the application.

## 1. `app` Directory

The `app` directory is the core of the Next.js application, containing the main layout, pages, and routing logic.

### `app/layout.tsx`

This file defines the root layout for the entire application. It wraps all pages with a common structure, including the HTML shell, body, and any global components.

```tsx
import "./globals.css";
import { GeistSans } from "geist/font/sans";
import { Toaster } from "sonner";
import { cn } from "@/lib/utils";
import { Navbar } from "@/components/navbar";

export const metadata = {
  title: "AI SDK Python Streaming Preview",
  description:
    "Use the Data Stream Protocol to stream chat completions from a Python endpoint (FastAPI) and display them using the useChat hook in your Next.js application.",
  openGraph: {
    images: [
      {
        url: "/og?title=AI SDK Python Streaming Preview",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    images: [
      {
        url: "/og?title=AI SDK Python Streaming Preview",
      },
    ],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head></head>
      <body className={cn(GeistSans.className, "antialiased dark")}>
        <Toaster position="top-center" richColors />
        <Navbar />
        {children}
      </body>
    </html>
  );
}
```

**Explanation:**

*   **Global Styles**: It imports the global stylesheet `globals.css`.
*   **Font**: It uses `GeistSans` for a consistent font across the application.
*   **Metadata**: It exports a `metadata` object, which Next.js uses to set the page's title and description for SEO and social media sharing. It also defines the Open Graph and Twitter card images.
*   **Root Layout Component**: The `RootLayout` component is the main layout.
    *   It renders the basic HTML structure (`<html>`, `<head>`, `<body>`).
    *   **`Toaster`**: It includes the `Toaster` component from the `sonner` library, which is used for displaying toast notifications.
    *   **`Navbar`**: It renders the main navigation bar, which will be present on all pages.
    *   **`children`**: It renders the `children` prop, which will be the content of the active page.
    *   **Styling**: It uses the `cn` utility from `@/lib/utils` to combine class names, applying the `GeistSans` font and a dark theme with anti-aliasing. 

### `app/(chat)/page.tsx`

This is the main page for the chat interface. It's a client component (`"use client"`) that brings together all the pieces of the chat functionality.

```tsx
// app/(chat)/page.tsx
"use client";

import { useEffect, useState, useMemo } from 'react';
import { useChat, Message as VercelAIMessage, CreateMessage } from 'ai/react'; // Added CreateMessage
import { Bungee } from 'next/font/google';
import CharacterSelector from "@/components/chat/CharacterSelector";
import ChatArea from "@/components/chat/ChatArea";
import { MultimodalInput } from "@/components/multimodal-input"; // ADDED
import { Message as CustomMessage } from '@/components/chat/ChatMessage';

const bungee = Bungee({ subsets: ['latin'], weight: '400' });

// Define characters here to have a single source of truth
const characters = [
  { id: 'chandler', name: 'Chandler Bing', avatarUrl: '/characters/chandler/avatar.png', greeting: "Could I BE any more ready to chat?", backgroundUrl: '/characters/chandler/background.png' },
  { id: 'tyrion', name: 'Tyrion Lannister', avatarUrl: '/characters/tyrion/avatar.png', greeting: "I drink and I know things. What do you want to know?", backgroundUrl: '/characters/tyrion/background.png' },
  { id: 'heisenberg', name: 'Walter Heisenberg', avatarUrl: '/characters/heisenberg/avatar.png', greeting: "I am the one who knocks. What do you want?", backgroundUrl: '/characters/heisenberg/background.png' },
  { id: 'po', name: 'Po the Panda', avatarUrl: `/characters/po/avatar.png?t=${new Date().getTime()}`, greeting: "Skadoosh! Ready for a chat?", backgroundUrl: '/characters/po/background.png' },
  { id: 'sherlock', name: 'Sherlock Holmes', avatarUrl: '/characters/sherlock/avatar.png', greeting: "The game is afoot. What is your query?", backgroundUrl: '/characters/sherlock/background.png' },
  { id: 'yoda', name: 'Yoda', avatarUrl: '/characters/yoda/avatar.png', greeting: "A question, you have. Ask, you must.", backgroundUrl: '/characters/yoda/background.png' },
  { id: 'michael', name: 'Michael Scott', avatarUrl: '/characters/michael/avatar.png', greeting: "Well, well, well, how the turntables... So, what can I do for you?", backgroundUrl: '/characters/michael/background.png' },
];

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [selectedCharId, setSelectedCharId] = useState<string>('chandler');

  useEffect(() => {
    const getOrCreateSessionId = (): string => {
      let id = sessionStorage.getItem('chatSessionId');
      if (!id) {
        id = crypto.randomUUID();
        sessionStorage.setItem('chatSessionId', id);
        console.log('New session ID created and stored:', id);
      } else {
        console.log('Existing session ID retrieved:', id);
      }
      return id;
    };
    if (typeof window !== 'undefined') {
      setSessionId(getOrCreateSessionId());
    }
  }, []);

  const selectedCharacter = useMemo(() => characters.find(c => c.id === selectedCharId), [selectedCharId]);

  const {
    messages: vercelMessages,
    input,
    handleInputChange,
    setInput,
    handleSubmit: originalHandleSubmit,
    isLoading,
    append,
    setMessages,
    stop
  } = useChat({
    api: '/api/v1/chat',
    initialMessages: [
      {
        id: `init_greet_${selectedCharId}_0`,
        role: 'assistant',
        content: selectedCharacter?.greeting || "Hello.",
      },
    ],
    body: {
      session_id: sessionId,
      character_id: selectedCharId,
    },
    onFinish: () => {
      // You can add logic here if needed when a response is fully received
    }
  });

  useEffect(() => {
    if (selectedCharacter) {
      setMessages([
        {
          id: `init_greet_${selectedCharacter.id}_0`,
          role: 'assistant',
          content: selectedCharacter.greeting,
        },
      ]);
    }
  }, [selectedCharId, selectedCharacter, setMessages]);


  const adaptedMessages: CustomMessage[] = useMemo(() => {
    return vercelMessages.map((msg: VercelAIMessage): CustomMessage => ({
      id: msg.id,
      text: msg.content,
      sender: msg.role === 'user' ? 'user' : 'character',
      characterName: msg.role === 'assistant' ? (selectedCharacter?.name || 'Character') : undefined,
      avatarUrl: msg.role === 'assistant' ? (selectedCharacter?.avatarUrl) : undefined,
    }));
  }, [vercelMessages, selectedCharacter]);

  return (
    <div className="flex flex-col h-screen bg-slate-100 dark:bg-neutral-900">
      <main className="flex-grow flex flex-col overflow-y-auto p-4">
        {/* This div constrains the width of CharacterSelector and ChatArea */}
        <div className="w-full md:max-w-3xl mx-auto flex flex-col flex-grow space-y-4">
          <CharacterSelector characters={characters} selectedCharacterId={selectedCharId} onCharacterSelect={setSelectedCharId} />
          <ChatArea messages={adaptedMessages} isLoading={isLoading} error={null} character={selectedCharacter} />
        </div>
      </main>
      {/* Form structure adapted from original components/chat.tsx, wraps MultimodalInput */}
      <form
        onSubmit={(e) => {
          e.preventDefault(); // Prevent default form submission by browser
          if (!isLoading && (input.trim() || sessionId)) { // Check if not loading and input is present or sessionID to allow submit
            originalHandleSubmit(e); // Call useChat's handleSubmit
          }
        }}
        className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl sticky bottom-0 z-10"
      >
        {sessionId && ( // Ensure sessionId is available before rendering MultimodalInput
          <MultimodalInput
            chatId={sessionId} // Using dynamic sessionId
            input={input}
            setInput={setInput} // CHANGED: Use setInput from useChat directly
            isLoading={isLoading}
            stop={stop}
            messages={vercelMessages} // Pass raw Vercel AI messages
            setMessages={setMessages}
            append={append}
            handleSubmit={(e) => { // Pass a handler that calls originalHandleSubmit
              if (!isLoading && (input.trim() || sessionId)) {
                originalHandleSubmit(e as React.FormEvent<HTMLFormElement>);
              }
            }}
          />
        )}
      </form>
    </div>
  );
}
```

**Explanation:**

*   **State Management**:
    *   It uses `useState` to manage the `sessionId` and the `selectedCharId`.
    *   The `useEffect` hook is used to get or create a unique `sessionId` for the chat session and store it in `sessionStorage`. This ensures that the chat history is preserved across page reloads.
*   **`useChat` Hook**:
    *   This is the core of the chat functionality, provided by the Vercel AI SDK (`ai/react`).
    *   `api`: It's configured to send requests to the `/api/v1/chat` backend endpoint.
    *   `initialMessages`: It sets the initial greeting message from the selected character.
    *   `body`: It sends the `session_id` and `character_id` with each request, allowing the backend to maintain context.
    *   It provides all the necessary state and functions to manage the chat: `messages`, `input`, `handleInputChange`, `handleSubmit`, `isLoading`, etc.
*   **Character Data**: It defines an array of `characters` with their details (ID, name, avatar, greeting).
*   **Message Adaptation**:
    *   The `useMemo` hook is used to adapt the messages from the format used by the `ai/react` library to the format expected by the custom `ChatMessage` component. This is a good practice for decoupling the UI from the data source.
*   **UI Composition**:
    *   The component renders the `CharacterSelector`, allowing the user to choose a character.
    *   The `ChatArea` is rendered to display the conversation.
    *   The `MultimodalInput` component is used for user input, including text and potentially other media.
*   **Form Handling**:
    *   It wraps the `MultimodalInput` in a `<form>` element and uses the `onSubmit` handler to trigger the `handleSubmit` function from the `useChat` hook. This ensures that the message is sent when the user submits the form (e.g., by pressing Enter).

### `app/og/route.tsx`

This file is a Next.js Route Handler that dynamically generates Open Graph (OG) images for social media sharing. When a link to the application is shared on platforms like Twitter or Facebook, this route is called to create a custom preview image.

```tsx
/* eslint-disable @next/next/no-img-element */

import { ImageResponse } from "next/server";

export const runtime = "edge";
export const preferredRegion = ["iad1"];

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  const title = searchParams.get("title");
  const description = searchParams.get("description");

  const imageData = await fetch(
    new URL("./background.png", import.meta.url)
  ).then((res) => res.arrayBuffer());

  const geistSemibold = await fetch(
    new URL("../../assets/geist-semibold.ttf", import.meta.url)
  ).then((res) => res.arrayBuffer());

  return new ImageResponse(
    (
      <div
        tw="flex h-full w-full bg-black"
        style={{ fontFamily: "Geist Sans" }}
      >
        {/* @ts-expect-error */}
        <img src={imageData} alt="vercel opengraph background" />
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            height: "100%",
            width: "100%",
            padding: "64px",
          }}
        >
          <div
            tw="text-zinc-50 tracking-tight flex-grow-1 flex flex-col justify-center leading-[1.1]"
            style={{
              fontWeight: 500,
              fontSize: 80,
              color: "black",
              letterSpacing: "-0.05em",
            }}
          >
            {title}
          </div>
          <div tw="text-[40px]" style={{ color: "#7D7D7D" }}>
            {description}
          </div>
        </div>
      </div>
    ),
    {
      width: 1200,
      height: 628,
      fonts: [
        {
          name: "geist",
          data: geistSemibold,
          style: "normal",
        },
      ],
    }
  );
}
```

**Explanation:**

*   **Runtime**: `export const runtime = "edge";` specifies that this route should run on the Edge Runtime, which is optimized for performance and low latency.
*   **`GET` Handler**: The `GET` function is the main handler for this route.
    *   It retrieves the `title` and `description` from the URL's query parameters.
    *   It fetches the background image and the `Geist Sans` font.
*   **`ImageResponse`**:
    *   It uses the `ImageResponse` class from `next/server` to generate an image from JSX and CSS-in-JS (using the `tw` prop for Tailwind-like syntax).
    *   The JSX defines the layout of the OG image, including the background image and the text for the title and description.
    *   The `width`, `height`, and `fonts` options are configured for the generated image.

## 2. `components` Directory

This directory contains all the reusable React components that make up the user interface.

### `components/navbar.tsx`

This component renders the main navigation bar at the top of the page.

```tsx
"use client";

import { Bungee } from 'next/font/google';
import { Button } from "./ui/button";
import { GitIcon } from "./icons";
import Link from "next/link";

const bungee = Bungee({ subsets: ['latin'], weight: '400' });

export const Navbar = () => {
  return (
    <nav className="p-2 flex flex-row items-center justify-between">
      <Link href="https://deepak-karkala.github.io/blog/">
        <Button size="sm">
          Project blog
        </Button>
      </Link>

      <h1 className={`${bungee.className} text-3xl font-bold [word-spacing:0.2em]`}>
        <span className="text-blue-500">Chat</span> <span className="text-green-500">with</span> <span className="text-red-500">Characters</span>
      </h1>

      <Link href="https://github.com/deepak-karkala/nextjs-fastapi-starter">
        <Button size="sm">
          <GitIcon />
          Code
        </Button>
      </Link>
    </nav>
  );
};
```

**Explanation:**

*   **Styling**: It uses Tailwind CSS classes for layout and styling, and the `Bungee` font for the main title.
*   **Links**:
    *   It includes a link to the project blog.
    *   The main title is a prominent feature with custom styling.
    *   It provides a link to the project's GitHub repository, complete with a Git icon.
*   **`Button` Component**: It uses the custom `Button` component from `components/ui/button` for the links, ensuring a consistent button style.
*   **`Link` Component**: It uses the Next.js `Link` component for client-side navigation, which is more performant than a standard `<a>` tag.

### `components/multimodal-input.tsx`

This component is a sophisticated text input for the chat interface, designed to handle user input, form submission, and other advanced features.

```tsx
"use client";

import type { ChatRequestOptions, CreateMessage, Message } from "ai";
import { motion } from "framer-motion";
import type React from "react";
import {
  useRef,
  useEffect,
  useCallback,
  type Dispatch,
  type SetStateAction,
} from "react";
import { toast } from "sonner";
import { useLocalStorage, useWindowSize } from "usehooks-ts";

import { cn, sanitizeUIMessages } from "@/lib/utils";

import { ArrowUpIcon, StopIcon } from "./icons";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";

const suggestedActions = [
  {
    title: "What is the weather",
    label: "in San Francisco?",
    action: "What is the weather in San Francisco?",
  },
  {
    title: "How is python useful",
    label: "for AI engineers?",
    action: "How is python useful for AI engineers?",
  },
];

export function MultimodalInput({
  chatId,
  input,
  setInput,
  isLoading,
  stop,
  messages,
  setMessages,
  append,
  handleSubmit,
  className,
}: {
  chatId: string;
  input: string;
  setInput: (value: string) => void;
  isLoading: boolean;
  stop: () => void;
  messages: Array<Message>;
  setMessages: Dispatch<SetStateAction<Array<Message>>>;
  append: (
    message: Message | CreateMessage,
    chatRequestOptions?: ChatRequestOptions,
  ) => Promise<string | null | undefined>;
  handleSubmit: (
    event?: {
      preventDefault?: () => void;
    },
    chatRequestOptions?: ChatRequestOptions,
  ) => void;
  className?: string;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { width } = useWindowSize();

  useEffect(() => {
    if (textareaRef.current) {
      adjustHeight();
    }
  }, []);

  const adjustHeight = () => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight + 2}px`;
    }
  };

  const [localStorageInput, setLocalStorageInput] = useLocalStorage(
    "input",
    "",
  );

  useEffect(() => {
    if (textareaRef.current) {
      const domValue = textareaRef.current.value;
      // Prefer DOM value over localStorage to handle hydration
      const finalValue = domValue || localStorageInput || "";
      setInput(finalValue);
      adjustHeight();
    }
    // Only run once after hydration
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setLocalStorageInput(input);
  }, [input, setLocalStorageInput]);

  const handleInput = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(event.target.value);
    adjustHeight();
  };

  const submitForm = useCallback(() => {
    handleSubmit(undefined, {});
    setLocalStorageInput("");

    if (width && width > 768) {
      textareaRef.current?.focus();
    }
  }, [handleSubmit, setLocalStorageInput, width]);

  return (
    <div className="relative w-full flex flex-col gap-4">
      {messages.length === 0 && (
        <div className="grid sm:grid-cols-2 gap-2 w-full">
          {suggestedActions.map((suggestedAction, index) => (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              transition={{ delay: 0.05 * index }}
              key={`suggested-action-${suggestedAction.title}-${index}`}
              className={index > 1 ? "hidden sm:block" : "block"}
            >
              <Button
                variant="ghost"
                onClick={async () => {
                  append({
                    role: "user",
                    content: suggestedAction.action,
                  });
                }}
                className="text-left border rounded-xl px-4 py-3.5 text-sm flex-1 gap-1 sm:flex-col w-full h-auto justify-start items-start"
              >
                <span className="font-medium">{suggestedAction.title}</span>
                <span className="text-muted-foreground">
                  {suggestedAction.label}
                </span>
              </Button>
            </motion.div>
          ))}
        </div>
      )}

      <Textarea
        ref={textareaRef}
        placeholder="Send a message..."
        value={input}
        onChange={handleInput}
        className={cn(
          "min-h-[24px] max-h-[calc(75dvh)] overflow-hidden resize-none rounded-xl !text-base bg-muted",
          className,
        )}
        rows={3}
        autoFocus
        data-testid="chat-input"
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();

            if (isLoading) {
              toast.error("Please wait for the model to finish its response!");
            } else {
              submitForm();
            }
          }
        }}
      />

      {isLoading ? (
        <Button
          className="rounded-full p-1.5 h-fit absolute bottom-2 right-2 m-0.5 border dark:border-zinc-600"
          onClick={(event) => {
            event.preventDefault();
            stop();
            setMessages((messages) => sanitizeUIMessages(messages));
          }}
        >
          <StopIcon size={14} />
        </Button>
      ) : (
        <Button
          className="rounded-full p-1.5 h-fit absolute bottom-2 right-2 m-0.5 border dark:border-zinc-600"
          data-testid="send-button"
          onClick={(event) => {
            event.preventDefault();
            submitForm();
          }}
          disabled={input.length === 0}
        >
          <ArrowUpIcon size={14} />
        </Button>
      )}
    </div>
  );
}
```

**Explanation:**

*   **Props**: It accepts a wide range of props from the `useChat` hook, including the current `input` value, `setInput` function, `isLoading` state, and `handleSubmit` function.
*   **Dynamic Textarea**: The `Textarea` component automatically adjusts its height based on the content, providing a better user experience than a fixed-size input field.
*   **Suggested Actions**: If there are no messages in the chat, it displays a list of suggested actions to help the user get started. These are animated with `framer-motion` for a smooth appearance.
*   **Local Storage**: It uses the `useLocalStorage` hook from `usehooks-ts` to persist the user's input across page reloads. This prevents the user from losing their drafted message if they accidentally navigate away.
*   **Submit and Stop Buttons**:
    *   When the chat is not loading, it displays a "Send" button (with an `ArrowUpIcon`) that is disabled if the input is empty.
    *   When the chat is loading, it displays a "Stop" button (with a `StopIcon`) that allows the user to interrupt the model's response.
*   **Keyboard Shortcuts**: It listens for the "Enter" key (without the Shift key) to submit the form, providing a familiar chat experience.
*   **Error Handling**: If the user tries to send a message while the model is already generating a response, it displays a toast notification using the `sonner` library. 

### `components/chat/CharacterSelector.tsx`

This component displays a horizontally scrollable list of characters that the user can choose to interact with.

```tsx
"use client";
import React, from 'react';
import Image from 'next/image';

interface Character {
  id: string;
  name: string;
  avatarUrl: string;
}

interface CharacterSelectorProps {
  characters: Character[];
  selectedCharacterId: string | null;
  onCharacterSelect: (id: string) => void;
}

const CharacterSelector: React.FC<CharacterSelectorProps> = ({ characters, selectedCharacterId, onCharacterSelect }) => {

  const handleCharacterSelect = (char: Character) => {
    onCharacterSelect(char.id);
  };

  return (
    <div className="mb-4">
      <h2 className="text-lg font-semibold mb-2 text-center text-slate-700 dark:text-neutral-200">Select a Character</h2>
      {/* Reduced mt-8 to mt-2 */}
      <div className="overflow-x-auto pb-2 mt-2">
        {/* Outer container for horizontal scrolling */}
        <div className="flex flex-nowrap justify-center space-x-2 px-1"> {/* justify-center, flex-nowrap, added px-1 for slight edge padding */}
          {characters.map((char) => {
            const isSelected = selectedCharacterId === char.id;

            return (
              // Each character item wrapper - ensure it doesn't shrink
              <div
                key={char.id}
                className={`
                  flex-shrink-0 p-1 rounded-lg transition-all duration-200 ease-in-out transform text-center 
                  cursor-pointer
                  ${isSelected
                    ? 'ring-2 ring-blue-500'
                    : 'opacity-70 hover:opacity-100'
                  }
                `}
                onClick={() => handleCharacterSelect(char)}
                role="button"
                tabIndex={0}
                aria-pressed={isSelected}
                aria-label={`Select ${char.name}`}
              >
                <Image
                  src={char.avatarUrl}
                  alt={char.name}
                  width={60}
                  height={60}
                  className={`rounded-full mx-auto`}
                  loading="eager"
                />
                <p className="mt-1 text-xs text-slate-600 dark:text-neutral-300 w-24 break-words">{char.name}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
export default CharacterSelector;
```

**Explanation:**

*   **Props**: It takes a list of `characters`, the `selectedCharacterId`, and an `onCharacterSelect` callback function as props.
*   **Horizontal Scrolling**: The component is designed to be horizontally scrollable (`overflow-x-auto`), which is useful for displaying a large number of characters without taking up too much vertical space.
*   **Character Rendering**: It maps over the `characters` array and renders each character with their avatar and name.
*   **Selection State**:
    *   The currently selected character is highlighted with a blue ring, while the others are slightly faded out.
    *   It uses the `aria-pressed` attribute to indicate the selected state for accessibility.
*   **User Interaction**: When a user clicks on a character, the `onCharacterSelect` callback is called with the character's ID, which updates the state in the parent component (`ChatPage`).
*   **`Image` Component**: It uses the Next.js `Image` component for optimized image loading. The `loading="eager"` prop suggests that the browser should prioritize loading these images. 

### `components/chat/ChatArea.tsx`

This component is responsible for rendering the main chat conversation area, including the messages, typing indicator, and any errors.

```tsx
"use client";
import React, { useEffect, useRef } from 'react';
import ChatMessage, { Message } from './ChatMessage';
// import { Message as VercelAIMessage } from 'ai/react'; // Assuming this is the type for Vercel AI SDK messages

interface Character {
  id: string;
  name: string;
  avatarUrl: string;
  greeting: string;
  backgroundUrl: string;
}

interface ChatAreaProps {
  messages: Message[];
  isLoading?: boolean;
  error?: string | null;
  character?: Character;
}

const ChatArea: React.FC<ChatAreaProps> = ({ messages, isLoading, error, character }) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const backgroundStyle = {
    backgroundImage: `url(${character?.backgroundUrl || '/characters/chandler/background.png'})`,
  };

  return (
    <div
      ref={containerRef}
      className="flex-grow overflow-y-auto mb-4 bg-cover bg-center p-4 rounded-lg"
      style={backgroundStyle}
    >
      {messages.length === 0 && !isLoading ? (
        <div className="flex justify-center items-center h-full">
          <p className="text-slate-500 dark:text-neutral-400"> No messages yet. Say hi to Chandler! </p>
        </div>
      ) : (
        <div className="space-y-4 px-12">
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}
      {isLoading && character && (
        <div data-testid="typing-indicator" className="flex justify-center items-center py-2">
          <p className="text-slate-500 dark:text-neutral-400"> {character.name} is typing... </p>
        </div>
      )}
      {error && (
        <div className="flex justify-center items-center py-2">
          <p className="text-red-500 dark:text-red-400">{error}</p>
        </div>
      )}
    </div>
  );
};

export default ChatArea;
```

**Explanation:**

*   **Props**: It accepts `messages`, `isLoading`, `error`, and `character` data to render the chat state.
*   **Auto-Scrolling**: It uses a `useEffect` hook and a `ref` (`messagesEndRef`) to automatically scroll to the bottom of the chat area whenever a new message is added. This ensures the user always sees the latest message.
*   **Dynamic Background**: The background of the chat area changes based on the selected `character`.
*   **Conditional Rendering**:
    *   If there are no messages and the chat is not loading, it displays a welcome message.
    *   It maps over the `messages` array and renders a `ChatMessage` component for each message.
    *   If `isLoading` is true, it displays a "typing..." indicator with the character's name.
    *   If there is an `error`, it displays the error message in red.

*   **`Image` Component**: It uses the Next.js `Image` component for optimized image loading. The `loading="eager"` prop suggests that the browser should prioritize loading these images. 

### `components/chat/ChatMessage.tsx`

This component is responsible for rendering a single message in the chat, with different styles for user messages and character messages.

```tsx
import React from 'react';
import Image from 'next/image'; // Import Next.js Image component

export interface Message {
  id: string;
  text: string;
  sender: 'user' | 'character';
  characterName?: string; // Optional, only for character messages
  imageUrl?: string; // Optional, for user messages with images
  avatarUrl?: string; // Added for avatar
}

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.sender === 'user';
  const userAvatarUrl = '/characters/user/avatar.png'; // Define a default user avatar

  const bubbleClasses = isUser
    ? 'bg-blue-500 text-white message-bubble-user' // User messages remain blue
    : 'bg-gray-300 dark:bg-neutral-700 text-gray-900 dark:text-neutral-100 message-bubble-character'; // Character messages updated

  // Avatar component
  const Avatar = ({ src, alt }: { src: string; alt: string }) => (
    <Image
      src={src}
      alt={alt}
      width={32} // sm: 40
      height={32} // sm: 40
      className="rounded-full"
      priority // Eager load avatars as they are likely in viewport
    />
  );

  const messageContent = (
    <div
      className={`
        p-3 rounded-lg break-words
        max-w-xs sm:max-w-md md:max-w-lg lg:max-w-xl xl:max-w-2xl // Adjusted max-widths slightly
        ${bubbleClasses}
        shadow-lg // Added shadow for floating effect
      `}
    >
      {!isUser && message.characterName && (
        <p className="text-xs font-semibold mb-1" data-testid="character-name">
          {message.characterName}
        </p>
      )}
      {message.imageUrl && (
        <div className="my-2">
          <img
            src={message.imageUrl}
            alt="Uploaded content"
            className="rounded-md max-w-xs max-h-64 object-contain"
          />
        </div>
      )}
      <p className="whitespace-pre-wrap">{message.text}</p>
    </div>
  );

  return (
    <div
      className={`flex items-end mb-3 gap-2 ${isUser ? 'justify-end' : 'justify-start'}`}
      data-testid="message-container"
    >
      {!isUser && message.avatarUrl && (
        <div className="flex-shrink-0">
          <Avatar src={message.avatarUrl} alt={message.characterName || 'Character'} />
        </div>
      )}
      {messageContent}
      {isUser && (
        <div className="flex-shrink-0">
          <Avatar src={userAvatarUrl} alt="User" />
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
```

**Explanation:**

*   **`Message` Interface**: It defines a `Message` interface that represents the structure of a single chat message.
*   **User vs. Character Styling**:
    *   It checks if the `sender` is a `'user'` or a `'character'` and applies different styling accordingly.
    *   User messages are aligned to the right, and character messages are aligned to the left.
    *   The message bubbles have different background colors to distinguish between the user and the character.
*   **Avatars**: It displays an avatar for both the user and the character. The character's avatar is provided in the `message` object, while the user's avatar is a default image.
*   **Image Handling**: If a message includes an `imageUrl`, it renders the image within the message bubble.
*   **Responsive Design**: The `max-w-*` classes ensure that the message bubbles are responsive and don't take up the full width of the screen on larger devices.
*   **Whitespace**: `whitespace-pre-wrap` is used to preserve newlines and spaces in the message text.

### `components/chat/MessageInput.tsx`

This component provides a rich text input field for the user to type and send messages. It also includes functionality for attaching images.

```tsx
"use client";
import React from 'react';

// Icon definitions (defined once)
const PaperclipIconSvg = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M18.375 12.739l-7.693 7.693a4.5 4.5 0 01-6.364-6.364l10.94-10.94A3.375 3.375 0 1112.81 8.42l-7.693 7.693a.375.375 0 01-.53-.53l7.693-7.693a2.25 2.25 0 00-3.182-3.182L4.929 15.929a4.5 4.5 0 006.364 6.364l7.693-7.693a.375.375 0 01.53.53z" />
    </svg>
);
PaperclipIconSvg.displayName = 'PaperclipIcon';
const PaperAirplaneIconSvg = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
    </svg>
);
PaperAirplaneIconSvg.displayName = 'PaperAirplaneIcon';

const XCircleIconSvg = (props: React.SVGProps<SVGSVGElement>) => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" {...props}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);
XCircleIconSvg.displayName = 'XCircleIcon';

interface MessageInputProps {
    input: string;
    handleInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
    handleSubmit: (e: React.FormEvent<HTMLFormElement>) => void;
    handleFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    imagePreviewUrl: string | null;
    onRemoveImage: () => void;
    isLoading?: boolean; // Added isLoading prop
}

const MessageInput: React.FC<MessageInputProps> = ({
    input,
    handleInputChange,
    handleSubmit,
    handleFileChange,
    imagePreviewUrl,
    onRemoveImage,
    isLoading // Destructure isLoading
}) => {
    const textareaRef = React.useRef<HTMLTextAreaElement>(null);
    const fileInputRef = React.useRef<HTMLInputElement>(null);

    // Auto-resize textarea
    React.useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = '0px';
            const scrollHeight = textareaRef.current.scrollHeight;
            textareaRef.current.style.height = scrollHeight + 'px';
        }
    }, [input]);

    // Handle keyboard events
    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey && !imagePreviewUrl) {
            e.preventDefault();
            const form = e.currentTarget.closest('form');
            if (form && input.trim()) {
                const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
                form.dispatchEvent(submitEvent);
            }
        } else if (e.key === 'Enter' && !e.shiftKey && imagePreviewUrl && !input.trim()) {
            e.preventDefault();
            const form = e.currentTarget.closest('form');
            if (form) {
                const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
                form.dispatchEvent(submitEvent);
            }
        }
    };

    const triggerFileInput = () => {
        fileInputRef.current?.click();
    };

    const isSubmitDisabled = (!input.trim() && !imagePreviewUrl) || isLoading;

    return (
        <div className="">
            {imagePreviewUrl && (
                <div className="mb-2 relative group w-24 h-24 border border-slate-300 dark:border-slate-600 rounded-md overflow-hidden">
                    <img
                        src={imagePreviewUrl}
                        alt="Uploaded image"
                        data-testid="uploaded-image-preview"
                        className="w-full h-full object-cover"
                    />
                    <button
                        type="button"
                        onClick={onRemoveImage}
                        aria-label="Remove image"
                        className="absolute top-1 right-1 bg-black/50 text-white rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                        <XCircleIconSvg className="w-5 h-5" />
                    </button>
                </div>
            )}

            <form
                data-testid="message-form"
                onSubmit={handleSubmit}
            >
                <div className="flex items-end space-x-2 md:space-x-3">
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        className="hidden"
                        accept="image/*"
                        data-testid="file-input"
                    />
                    <button
                        type="button"
                        onClick={triggerFileInput}
                        aria-label="Upload image"
                        className="p-2 text-slate-500 dark:text-neutral-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
                    >
                        <PaperclipIconSvg className="w-5 h-5 md:w-6 md:h-6" />
                    </button>

                    <textarea
                        ref={textareaRef}
                        id="message-input-textarea"
                        value={input}
                        onChange={handleInputChange}
                        onKeyDown={handleKeyDown}
                        placeholder="Type your message..."
                        rows={1}
                        className="flex-grow p-2.5 text-sm md:text-base border-slate-300 dark:border-slate-600 rounded-lg focus:ring-blue-500 focus:border-blue-500 resize-none bg-slate-50 dark:bg-neutral-700 text-slate-900 dark:text-neutral-100 dark:placeholder-neutral-400"
                        style={{ overflowY: 'hidden' }}
                        disabled={isLoading} // Disable textarea when loading
                    />

                    <button
                        type="submit"
                        aria-label="Send message"
                        disabled={isSubmitDisabled}
                        className="p-2.5 text-white bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50"
                    >
                        <PaperAirplaneIconSvg className="w-5 h-5 md:w-6 md:h-6" />
                    </button>
                </div>
            </form>
        </div>
    );
};

export default MessageInput;
```

**Explanation:**

*   **Icons**: It defines several SVG icons as React components for use in the input field.
*   **Auto-Resizing Textarea**: The `textarea` automatically resizes vertically as the user types, thanks to the `useEffect` hook that adjusts its height based on `scrollHeight`.
*   **File Upload**:
    *   It includes a hidden file input (`<input type="file">`) for image uploads.
    *   A "paperclip" button triggers the file input when clicked.
    *   When an image is selected, it displays a preview of the image with a button to remove it.
*   **Submit Logic**:
    *   The submit button is disabled if there is no text and no image, or if the chat is currently loading a response.
    *   The `handleKeyDown` function allows the user to submit the form by pressing "Enter" (without "Shift").
*   **Loading State**: The `textarea` and submit button are disabled when `isLoading` is true, preventing the user from sending multiple messages while waiting for a response.

### `components/ui/button.tsx`

This component is a reusable and highly customizable button, built using `class-variance-authority` (CVA). It serves as a great example of how to create variant-based UI components.

```tsx
import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground shadow-sm hover:bg-destructive/90",
        outline:
          "border border-input bg-background shadow-sm hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground shadow-sm hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-md px-8",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
```

**Explanation:**

*   **`class-variance-authority` (CVA)**:
    *   This library is used to define a set of button styles with different variants.
    *   The `buttonVariants` object defines the base classes for all buttons, as well as several variants for `variant` (e.g., `default`, `destructive`, `outline`) and `size` (e.g., `sm`, `lg`).
*   **`Slot` Component**:
    *   The `asChild` prop, combined with the `Slot` component from `@radix-ui/react-slot`, allows this button component to "slot" its props onto its direct child.
    *   This is incredibly useful for cases where you need to wrap another component (like a Next.js `Link`) with the button's styling and behavior, without creating an extra DOM element.
*   **`cn` Utility**: The `cn` utility from `@/lib/utils` is used to merge the CVA variants with any additional class names passed to the component.
*   **`React.forwardRef`**: This allows parent components to pass a `ref` down to the underlying `<button>` element, which is useful for focusing the button or accessing its DOM node.

### `components/ui/textarea.tsx`

This component is a simple, styled textarea that can be used throughout the application.

```tsx
import * as React from "react"

import { cn } from "@/lib/utils"

const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.ComponentProps<"textarea">
>(({ className, ...props }, ref) => {
  return (
    <textarea
      className={cn(
        "flex min-h-[60px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-base shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 md:text-sm",
        className
      )}
      ref={ref}
      {...props}
    />
  )
})
Textarea.displayName = "Textarea"

export { Textarea }
```

**Explanation:**

*   **Styling**: It uses Tailwind CSS classes to provide a consistent look and feel for textareas, including styles for focus, disabled states, and placeholder text.
*   **`cn` Utility**: The `cn` utility allows for easy customization by merging the default classes with any additional classes passed in the `className` prop.
*   **`React.forwardRef`**: Just like the `Button` component, it uses `React.forwardRef` to allow parent components to get a `ref` to the underlying `<textarea>` element. This is crucial for the auto-resizing functionality in `MessageInput.tsx`.

### `components/icons.tsx`

This file is a library of SVG icons that are used throughout the application. Each icon is a small, reusable React component.

```tsx
// A few representative examples from components/icons.tsx

export const BotIcon = () => {
  return (
    <svg
      height="16"
      strokeLinejoin="round"
      viewBox="0 0 16 16"
      width="16"
      style={{ color: "currentcolor" }}
    >
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M8.75 2.79933C9.19835 2.53997 9.5 2.05521 9.5 1.5C9.5 0.671573 8.82843 0 8 0C7.17157 0 6.5 0.671573 6.5 1.5C6.5 2.05521 6.80165 2.53997 7.25 2.79933V5H7C4.027 5 1.55904 7.16229 1.08296 10H0V13H1V14.5V16H2.5H13.5H15V14.5V13H16V10H14.917C14.441 7.16229 11.973 5 9 5H8.75V2.79933ZM7 6.5C4.51472 6.5 2.5 8.51472 2.5 11V14.5H13.5V11C13.5 8.51472 11.4853 6.5 9 6.5H7ZM7.25 11.25C7.25 12.2165 6.4665 13 5.5 13C4.5335 13 3.75 12.2165 3.75 11.25C3.75 10.2835 4.5335 9.5 5.5 9.5C6.4665 9.5 7.25 10.2835 7.25 11.25ZM10.5 13C11.4665 13 12.25 12.2165 12.25 11.25C12.25 10.2835 11.4665 9.5 10.5 9.5C9.5335 9.5 8.75 10.2835 8.75 11.25C8.75 12.2165 9.5335 13 10.5 13Z"
        fill="currentColor"
      />
    </svg>
  );
};

export const GitIcon = () => {
  return (
    <svg
      height="16"
      strokeLinejoin="round"
      viewBox="0 0 16 16"
      width="16"
      style={{ color: "currentcolor" }}
    >
      <g clipPath="url(#clip0_872_3147)">
        <path
          fillRule="evenodd"
          clipRule="evenodd"
          d="M8 0C3.58 0 0 3.57879 0 7.99729C0 11.5361 2.29 14.5251 5.47 15.5847C5.87 15.6547 6.02 15.4148 6.02 15.2049C6.02 15.0149 6.01 14.3851 6.01 13.7154C4 14.0852 3.48 13.2255 3.32 12.7757C3.23 12.5458 2.84 11.836 2.5 11.6461C2.22 11.4961 1.82 11.1262 2.49 11.1162C3.12 11.1062 3.57 11.696 3.72 11.936C4.44 13.1455 5.59 12.8057 6.05 12.5957C6.12 12.0759 6.33 11.726 6.56 11.5261C4.78 11.3262 2.92 10.6364 2.92 7.57743C2.92 6.70773 3.23 5.98797 3.74 5.42816C3.66 5.22823 3.38 4.40851 3.82 3.30888C3.82 3.30888 4.49 3.09895 6.02 4.1286C6.66 3.94866 7.34 3.85869 8.02 3.85869C8.7 3.85869 9.38 3.94866 10.02 4.1286C11.55 3.08895 12.22 3.30888 12.22 3.30888C12.66 4.40851 12.38 5.22823 12.3 5.42816C12.81 5.98797 13.12 6.69773 13.12 7.57743C13.12 10.6464 11.25 11.3262 9.47 11.5261C9.76 11.776 10.01 12.2558 10.01 13.0056C10.01 14.0752 10 14.9349 10 15.2049C10 15.4148 10.15 15.6647 10.55 15.5847C12.1381 15.0488 13.5182 14.0284 14.4958 12.6673C15.4735 11.3062 15.9996 9.67293 16 7.99729C16 3.57879 12.42 0 8 0Z"
          fill="currentColor"
        />
      </g>
      <defs>
        <clipPath id="clip0_872_3147">
          <rect width="16" height="16" fill="white" />
        </clipPath>
      </defs>
    </svg>
  );
};

export const ArrowUpIcon = ({ size = 16 }: { size?: number }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </svg>
);
```

**Explanation:**

*   **Centralized Icons**: This file acts as a single source of truth for all icons, making them easy to manage and reuse.
*   **SVG as Components**: Each icon is a React component that returns an `<svg>` element. This is a common and efficient way to handle icons in a React application.
*   **Customization**: Most icons accept a `size` prop, allowing them to be rendered at different sizes. The `style={{ color: "currentcolor" }}` and `fill="currentColor"` attributes allow the icon's color to be controlled by the parent element's CSS `color` property.
*   **Readability**: The file is quite large, but it's well-organized, with each icon defined as a separate, named component.

### `components/chat.tsx`

This component is another implementation of the main chat interface. It integrates the `useChat` hook, message display, and the multimodal input field. It appears to be a more streamlined version compared to `app/(chat)/page.tsx`, without the character selection feature.

```tsx
"use client";

import { PreviewMessage, ThinkingMessage } from "@/components/message";
import { MultimodalInput } from "@/components/multimodal-input";
import { Overview } from "@/components/overview";
import { useScrollToBottom } from "@/hooks/use-scroll-to-bottom";
import { ToolInvocation } from "ai";
import { useChat } from "ai/react";
import { toast } from "sonner";

export function Chat() {
  const chatId = "001";

  const {
    messages,
    setMessages,
    handleSubmit,
    input,
    setInput,
    append,
    isLoading,
    stop,
  } = useChat({
    maxSteps: 4,
    onError: (error) => {
      if (error.message.includes("Too many requests")) {
        toast.error(
          "You are sending too many messages. Please try again later.",
        );
      }
    },
  });

  const [messagesContainerRef, messagesEndRef] =
    useScrollToBottom<HTMLDivElement>();

  return (
    <div className="flex flex-col min-w-0 h-[calc(100dvh-52px)] bg-background">
      <div
        ref={messagesContainerRef}
        className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4"
      >
        {messages.length === 0 && <Overview />}

        {messages.map((message, index) => (
          <PreviewMessage
            key={message.id}
            chatId={chatId}
            message={message}
            isLoading={isLoading && messages.length - 1 === index}
          />
        ))}

        {isLoading &&
          messages.length > 0 &&
          messages[messages.length - 1].role === "user" && <ThinkingMessage />}

        <div
          ref={messagesEndRef}
          className="shrink-0 min-w-[24px] min-h-[24px]"
        />
      </div>

      <form className="flex mx-auto px-4 bg-background pb-4 md:pb-6 gap-2 w-full md:max-w-3xl">
        <MultimodalInput
          chatId={chatId}
          input={input}
          setInput={setInput}
          handleSubmit={handleSubmit}
          isLoading={isLoading}
          stop={stop}
          messages={messages}
          setMessages={setMessages}
          append={append}
        />
      </form>
    </div>
  );
}
```

**Explanation:**

*   **`useChat` Hook**: It initializes the `useChat` hook from the Vercel AI SDK.
    *   `maxSteps`: This option is set to `4`, which might be used to limit the number of tool invocations or steps in a chain-of-thought process.
    *   `onError`: It includes an error handler that displays a toast notification if the user sends too many requests.
*   **`useScrollToBottom` Hook**: It uses a custom hook to automatically scroll the message container to the bottom when new messages are added.
*   **Message Rendering**:
    *   If there are no messages, it displays an `Overview` component, which likely provides some introductory information or suggested prompts.
    *   It maps over the `messages` and renders a `PreviewMessage` for each one.
    *   It displays a `ThinkingMessage` while the AI is generating a response.
*   **`MultimodalInput`**: It uses the `MultimodalInput` component for user input, passing down all the necessary props from the `useChat` hook.
*   **Static `chatId`**: Unlike the `ChatPage` component, this one uses a hardcoded `chatId`.

### `components/overview.tsx`

This file contains the implementation of the `Overview` component, which is used to display a summary of the chat history or suggested prompts.

```tsx
"use client";

import { useChat } from "ai/react";
import { useScrollToBottom } from "@/hooks/use-scroll-to-bottom";
import { PreviewMessage, ThinkingMessage } from "@/components/message";

export function Overview() {
  const { messages } = useChat();
  const [messagesContainerRef, messagesEndRef] =
    useScrollToBottom<HTMLDivElement>();

  return (
    <div
      ref={messagesContainerRef}
      className="flex flex-col min-w-0 gap-6 flex-1 overflow-y-scroll pt-4"
    >
      {messages.length === 0 && <Overview />}

      {messages.map((message, index) => (
        <PreviewMessage
          key={message.id}
          chatId="001"
          message={message}
          isLoading={false}
        />
      ))}

      {messages.length > 0 && messages[messages.length - 1].role === "user" && <ThinkingMessage />}

      <div
        ref={messagesEndRef}
        className="shrink-0 min-w-[24px] min-h-[24px]"
      />
    </div>
  );
}
```

**Explanation:**

*   **`useChat` Hook**: It uses the `useChat` hook from the Vercel AI SDK to get the chat history.
*   **`useScrollToBottom` Hook**: It uses a custom hook to automatically scroll the message container to the bottom when new messages are added.
*   **Message Rendering**:
    *   If there are no messages, it displays an `Overview` component, which likely provides some introductory information or suggested prompts.
    *   It maps over the `messages` and renders a `PreviewMessage` for each one.
    *   It displays a `ThinkingMessage` while the AI is generating a response.
*   **Static `chatId`**: It uses a hardcoded `chatId` to ensure that the `Overview` component is always rendered with the same chat history.