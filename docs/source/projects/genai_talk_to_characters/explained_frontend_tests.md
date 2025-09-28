# Frontend Tests Explained

This document provides an explanation of the frontend tests for the application.

## 1. `MessageInput` Component Tests

These tests cover the `MessageInput` component, which is responsible for the user's text input and message sending.

### `components/chat/__tests__/MessageInput.test.tsx`

```tsx
import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import MessageInput from '../MessageInput'

describe('MessageInput', () => {
  const mockHandleInputChange = jest.fn()
  const mockHandleSubmit = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders input field and send button', () => {
    render(
      <MessageInput 
        input=""
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    expect(screen.getByPlaceholderText(/type your message/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /send message/i })).toBeInTheDocument()
  })

  it('handles text input changes', async () => {
    const user = userEvent.setup()
    render(
      <MessageInput 
        input=""
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const input = screen.getByPlaceholderText(/type your message/i)
    await user.type(input, 'Hello Chandler!')
    
    expect(mockHandleInputChange).toHaveBeenCalled()
  })

  it('disables send button when input is empty', () => {
    render(
      <MessageInput 
        input=""
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const sendButton = screen.getByRole('button', { name: /send message/i })
    expect(sendButton).toBeDisabled()
  })

  it('enables send button when input is not empty', () => {
    render(
      <MessageInput 
        input="Hello"
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const sendButton = screen.getByRole('button', { name: /send message/i })
    expect(sendButton).not.toBeDisabled()
  })

  it('handles form submission', async () => {
    render(
      <MessageInput 
        input="Hello Chandler!"
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const form = screen.getByTestId('message-form')
    fireEvent.submit(form)
    
    expect(mockHandleSubmit).toHaveBeenCalled()
  })

  it('handles Enter key press to submit', async () => {
    const user = userEvent.setup()
    render(
      <MessageInput 
        input="Hello"
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const textarea = screen.getByPlaceholderText(/type your message/i)
    await user.type(textarea, '{Enter}')
    
    expect(mockHandleSubmit).toHaveBeenCalled()
  })

  it('does not submit on Shift+Enter', async () => {
    const user = userEvent.setup()
    render(
      <MessageInput 
        input="Hello"
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const textarea = screen.getByPlaceholderText(/type your message/i)
    await user.type(textarea, '{Shift>}{Enter}{/Shift}')
    
    expect(mockHandleSubmit).not.toHaveBeenCalled()
  })

  it('handles very long input text', async () => {
    const longText = 'a'.repeat(1000)
    
    render(
      <MessageInput 
        input={longText}
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const textarea = screen.getByPlaceholderText(/type your message/i)
    expect(textarea).toHaveValue(longText)
    expect(textarea).toHaveClass('resize-none')
  })

  it('handles pasting text', async () => {
    const user = userEvent.setup()
    render(
      <MessageInput 
        input=""
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const textarea = screen.getByPlaceholderText(/type your message/i)
    await user.click(textarea)
    await user.paste('Pasted text')
    
    expect(mockHandleInputChange).toHaveBeenCalled()
  })

  it('handles rapid typing', async () => {
    render(
      <MessageInput 
        input=""
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const textarea = screen.getByPlaceholderText(/type your message/i)
    
    // Simulate rapid typing with fireEvent for better control
    fireEvent.change(textarea, { target: { value: 'F' } })
    fireEvent.change(textarea, { target: { value: 'Fa' } })
    fireEvent.change(textarea, { target: { value: 'Fas' } })
    fireEvent.change(textarea, { target: { value: 'Fast' } })
    
    expect(mockHandleInputChange).toHaveBeenCalled()
    expect(mockHandleInputChange.mock.calls.length).toBeGreaterThanOrEqual(4)
  })

  it('handles special characters input', async () => {
    render(
      <MessageInput 
        input=""
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const textarea = screen.getByPlaceholderText(/type your message/i)
    const specialChars = '!@#$%^&*()_+'
    
    fireEvent.change(textarea, { target: { value: specialChars } })
    expect(mockHandleInputChange).toHaveBeenCalled()
  })

  it('handles emoji input', async () => {
    render(
      <MessageInput 
        input=""
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const textarea = screen.getByPlaceholderText(/type your message/i)
    fireEvent.change(textarea, { target: { value: '👋 🎉 🎈' } })
    
    expect(mockHandleInputChange).toHaveBeenCalled()
  })

  it('maintains input state between renders', () => {
    const { rerender } = render(
      <MessageInput 
        input="Initial text"
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    const textarea = screen.getByPlaceholderText(/type your message/i)
    expect(textarea).toHaveValue('Initial text')

    rerender(
      <MessageInput 
        input="Updated text"
        handleInputChange={mockHandleInputChange}
        handleSubmit={mockHandleSubmit}
      />
    )

    expect(textarea).toHaveValue('Updated text')
  })

  it('handles form submission prevention', async () => {
    const mockPreventDefault = jest.fn()
    const customHandleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      mockPreventDefault()
      mockHandleSubmit(event)
    }
    
    render(
      <MessageInput 
        input="Test"
        handleInputChange={mockHandleInputChange}
        handleSubmit={customHandleSubmit}
      />
    )

    const form = screen.getByTestId('message-form')
    fireEvent.submit(form)
    
    expect(mockPreventDefault).toHaveBeenCalled()
    expect(mockHandleSubmit).toHaveBeenCalled()
  })
})
```

**Explanation:**

These tests ensure that the `MessageInput` component functions as expected. They cover:

*   **Rendering:** The component renders the text input field and the send button correctly.
*   **Input Handling:** It correctly handles user input, including typing, pasting, and special characters.
*   **Button State:** The send button is disabled when the input is empty and enabled when there is text.
*   **Form Submission:** The component correctly handles form submission when the send button is clicked or the Enter key is pressed. It also correctly prevents submission on Shift+Enter.
*   **State Management:** The input field's value is correctly maintained across re-renders.
*   **Edge Cases:** The tests also cover edge cases like very long input and rapid typing to ensure the component is robust.

## 2. `ChatFlow` Integration Tests

These tests cover the integration of the various chat components, ensuring they work together to create a seamless chat experience.

### `components/chat/__tests__/ChatFlow.integration.test.tsx`

```tsx
import React from 'react'
import { render, screen, fireEvent, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import CharacterSelector from '../CharacterSelector'
import ChatArea from '../ChatArea'
import MessageInput from '../MessageInput'
import { useChat, UseChatReturn } from '@/hooks/useChat'
//import { Message as VercelAIMessage } from 'ai'
import { Message as CustomMessage } from '../ChatMessage'

// Mock the useChat hook
jest.mock('@/hooks/useChat')

const mockUseChatDefaultValues: UseChatReturn = {
  messages: [],
  input: '',
  isLoading: false,
  error: null,
  handleInputChange: jest.fn(),
  handleSubmit: jest.fn(),
  selectedCharacter: {
    id: 'chandler',
    name: 'Chandler Bing',
    avatarUrl: '/characters/chandler/avatar.png',
    greeting: 'Could I BE any more excited to chat?',
    backgroundUrl: '/characters/chandler/background.png'
  }
}

// Helper function to set up the mock
const setupMockUseChat = (override: Partial<UseChatReturn>): UseChatReturn => {
  const mockValues = { ...mockUseChatDefaultValues, ...override };
  (useChat as jest.Mock).mockReturnValue(mockValues);
  return mockValues;
}

// Component to render for testing
const TestChatComponent: React.FC<UseChatReturn> = (props) => {
  const characters = props.selectedCharacter ? [props.selectedCharacter] : [];
  return (
    <div>
      <CharacterSelector
        characters={characters}
        selectedCharacterId={props.selectedCharacter?.id || null}
        onCharacterSelect={jest.fn()}
      />
      <ChatArea
        messages={props.messages}
        isLoading={props.isLoading}
        error={props.error}
        character={props.selectedCharacter || undefined}
      />
      <MessageInput
        input={props.input}
        handleInputChange={props.handleInputChange}
        handleSubmit={props.handleSubmit}
        handleFileChange={jest.fn()}
        imagePreviewUrl={null}
        onRemoveImage={jest.fn()}
      />
    </div>
  );
};

describe('Chat Flow Integration', () => {
  let currentMockState: UseChatReturn;

  beforeEach(() => {
    jest.clearAllMocks();
    currentMockState = setupMockUseChat({});
  })

  it('renders initial empty state correctly', () => {
    render(<TestChatComponent {...currentMockState} />);
    expect(screen.getByText(/no messages yet/i)).toBeInTheDocument();
    expect(screen.getByText(/chandler bing/i)).toBeInTheDocument();
  })

  it('handles complete message flow', async () => {
    const user = userEvent.setup();
    const { rerender } = render(<TestChatComponent {...currentMockState} />);

    // User types a message
    const inputElement = screen.getByPlaceholderText(/type your message/i);
    await act(async () => {
      await user.type(inputElement, 'Hello Chandler!');
      // Simulate input change in mock by updating the state that MessageInput uses
      // In a real scenario, useChat.handleInputChange would update useChat.input
      // For testing, we directly set the mock input and re-render.
      currentMockState = setupMockUseChat({ ...currentMockState, input: 'Hello Chandler!' });
    });
    rerender(<TestChatComponent {...currentMockState} />);
    expect(inputElement).toHaveValue('Hello Chandler!');

    // User submits the message
    const form = screen.getByTestId('message-form');
    await act(async () => {
      fireEvent.submit(form);
    });
    // handleSubmit in the mock is called
    expect(currentMockState.handleSubmit).toHaveBeenCalled();

    // Simulate response from the hook by updating messages and clearing input
    const updatedMessages: CustomMessage[] = [
      { id: '1', text: 'Hello Chandler!', sender: 'user' },
      {
        id: '2',
        text: 'Could I BE any more excited to chat?',
        sender: 'character',
        characterName: 'Chandler'
      }
    ];
    act(() => {
      currentMockState = setupMockUseChat({ ...currentMockState, messages: updatedMessages, input: '' });
    });
    rerender(<TestChatComponent {...currentMockState} />);

    // Verify messages are displayed and input is cleared
    expect(screen.getByText('Hello Chandler!')).toBeInTheDocument();
    expect(screen.getByText('Could I BE any more excited to chat?')).toBeInTheDocument();
    expect(inputElement).toHaveValue('');
  })

  it('handles loading states correctly', () => {
    // Mock messages should be in the CustomMessage format, as expected by ChatArea via the mock hook
    const mockInitialUserMessagesAsCustom: CustomMessage[] = [
      { id: '1', text: 'Hello', sender: 'user' },
    ];

    act(() => {
      currentMockState = setupMockUseChat({
        messages: mockInitialUserMessagesAsCustom, // Use CustomMessage[] for the mock
        isLoading: true,
        input: '',
      });
    });

    render(<TestChatComponent {...currentMockState} />);

    expect(screen.getByText(/chandler bing is typing.../i)).toBeInTheDocument();
    const sendButton = screen.getByRole('button', { name: /send message/i });
    expect(sendButton).toBeDisabled();
  })

  it('handles error states correctly', () => {
    act(() => {
      currentMockState = setupMockUseChat({
        ...currentMockState,
        error: 'Failed to send message',
        messages: [{ id: '1', text: 'Hello', sender: 'user' } as CustomMessage]
      });
    });
    render(<TestChatComponent {...currentMockState} />);

    expect(screen.getByText(/failed to send message/i)).toBeInTheDocument();
    const inputElement = screen.getByPlaceholderText(/type your message/i);
    expect(inputElement).not.toBeDisabled();
  })

  it('maintains conversation context', () => {
    const contextMessages: CustomMessage[] = [
      { id: '1', text: 'How are you?', sender: 'user' },
      {
        id: '2',
        text: "I'm doing great! Could I BE any better?",
        sender: 'character',
        characterName: 'Chandler'
      }
    ];
    act(() => {
      currentMockState = setupMockUseChat({ ...currentMockState, messages: contextMessages });
    });
    render(<TestChatComponent {...currentMockState} />);

    expect(screen.getByText('How are you?')).toBeInTheDocument();
    expect(screen.getByText("I'm doing great! Could I BE any better?")).toBeInTheDocument();
  })
})
```

**Explanation:**

These integration tests simulate a full chat conversation to ensure that all the chat components (`CharacterSelector`, `ChatArea`, and `MessageInput`) work together correctly. The tests use a mocked version of the `useChat` hook to control the state of the chat.

*   **Initial State:** It verifies that the chat interface renders correctly in its initial, empty state.
*   **Message Flow:** It simulates a user typing a message, submitting it, and receiving a response, ensuring that the messages are displayed correctly and the input field is cleared.
*   **Loading State:** It checks that the loading indicator is displayed correctly when the application is waiting for a response from the backend.
*   **Error State:** It ensures that error messages are displayed appropriately if the message fails to send.
*   **Conversation Context:** It verifies that the conversation history is maintained correctly between interactions.

## 3. `ChatArea` Component Tests

These tests focus on the `ChatArea` component, which is responsible for displaying the conversation, including user messages, character responses, and loading/error states.

### `components/chat/__tests__/ChatArea.test.tsx`

```tsx
import React from 'react'
import { render, screen } from '@testing-library/react'
import '@testing-library/jest-dom'
import ChatArea from '../ChatArea'
import { Message } from '../ChatMessage'

// Mock the scrollIntoView function
window.HTMLElement.prototype.scrollIntoView = jest.fn()

const mockCharacter = {
  id: 'chandler',
  name: 'Chandler',
  avatarUrl: '/characters/chandler/avatar.png',
  greeting: 'Could I BE any more excited to chat?',
  backgroundUrl: '/characters/chandler/background.png',
}

describe('ChatArea', () => {
  const mockMessages: Message[] = [
    {
      id: '1',
      text: 'Hi Chandler!',
      sender: 'user'
    },
    {
      id: '2',
      text: 'Could I BE any more excited to chat?',
      sender: 'character',
      characterName: 'Chandler'
    }
  ]

  it('renders empty state correctly', () => {
    render(<ChatArea messages={[]} character={mockCharacter} />)
    expect(screen.getByText(/no messages yet/i)).toBeInTheDocument()
  })

  it('renders messages correctly', () => {
    render(<ChatArea messages={mockMessages} character={mockCharacter} />)
    expect(screen.getByText('Hi Chandler!')).toBeInTheDocument()
    expect(screen.getByText('Could I BE any more excited to chat?')).toBeInTheDocument()
  })

  it('shows loading state', () => {
    render(<ChatArea messages={mockMessages} isLoading={true} character={mockCharacter} />)
    expect(screen.getByText(/chandler is typing.../i)).toBeInTheDocument()
  })

  it('shows error state', () => {
    render(
      <ChatArea
        messages={mockMessages}
        error="Failed to send message"
        character={mockCharacter}
      />
    )
    expect(screen.getByText('Failed to send message')).toBeInTheDocument()
  })

  it('maintains message order', () => {
    render(<ChatArea messages={mockMessages} character={mockCharacter} />)
    const messageContainers = screen.getAllByTestId('message-container')

    // First message should be from user
    expect(messageContainers[0]).toHaveTextContent('Hi Chandler!')

    // Second message should be from Chandler
    expect(messageContainers[1]).toHaveTextContent('Could I BE any more excited to chat?')
  })
})
```

**Explanation:**

These tests verify that the `ChatArea` component behaves as expected in different scenarios:

*   **Empty State:** It correctly displays a message when there are no messages in the chat.
*   **Message Rendering:** It renders the user's messages and the character's responses correctly.
*   **Loading and Error States:** It accurately shows a typing indicator when loading and displays an error message when something goes wrong.
*   **Message Order:** It ensures that the messages are displayed in the correct chronological order.

## 4. `CharacterSelector` Component Tests

These tests are for the `CharacterSelector` component, which allows users to choose a character to chat with.

### `components/chat/__tests__/CharacterSelector.test.tsx`

```tsx
import React from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import '@testing-library/jest-dom'
import CharacterSelector from '../CharacterSelector'

const mockCharacters = [
  { id: 'chandler', name: 'Chandler Bing', avatarUrl: '/characters/chandler.png' },
  { id: 'joker', name: 'Joker', avatarUrl: '/characters/joker.png' },
]

const mockOnSelect = jest.fn();

describe('CharacterSelector', () => {

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the component title', () => {
    render(
      <CharacterSelector
        characters={mockCharacters}
        selectedCharacterId="chandler"
        onCharacterSelect={mockOnSelect}
      />
    )
    expect(screen.getByText('Select a Character')).toBeInTheDocument()
  })

  it('renders all character options', () => {
    render(
      <CharacterSelector
        characters={mockCharacters}
        selectedCharacterId="chandler"
        onCharacterSelect={mockOnSelect}
      />
    );
    expect(screen.getByText('Chandler Bing')).toBeInTheDocument();
    expect(screen.getByAltText('Chandler Bing')).toBeInTheDocument();
    expect(screen.getByText('Joker')).toBeInTheDocument();
    expect(screen.getByAltText('Joker')).toBeInTheDocument();
  });


  it('has correct button attributes for selected character', () => {
    render(
      <CharacterSelector
        characters={mockCharacters}
        selectedCharacterId="chandler"
        onCharacterSelect={mockOnSelect}
      />
    );
    const button = screen.getByRole('button', { name: /select chandler bing/i });
    expect(button).toHaveAttribute('aria-pressed', 'true');
    expect(button).toHaveAttribute('tabIndex', '0');
  });

  it('applies correct styling for the selected character', () => {
    render(
      <CharacterSelector
        characters={mockCharacters}
        selectedCharacterId="chandler"
        onCharacterSelect={mockOnSelect}
      />
    );
    const button = screen.getByRole('button', { name: /select chandler bing/i });

    // Check for classes that indicate it's selected
    expect(button).toHaveClass('ring-2', 'ring-blue-500');
    expect(button).not.toHaveClass('opacity-70');
  });

  it('applies correct styling for a non-selected character', () => {
    render(
      <CharacterSelector
        characters={mockCharacters}
        selectedCharacterId="chandler"
        onCharacterSelect={mockOnSelect}
      />
    );
    const button = screen.getByRole('button', { name: /select joker/i });
    expect(button).not.toHaveClass('ring-2', 'ring-blue-500');
    expect(button).toHaveClass('opacity-70');
  });


  it('calls onCharacterSelect when a character is clicked', async () => {
    const user = userEvent.setup();
    render(
      <CharacterSelector
        characters={mockCharacters}
        selectedCharacterId="chandler"
        onCharacterSelect={mockOnSelect}
      />
    );
    const jokerButton = screen.getByRole('button', { name: /select joker/i });
    await user.click(jokerButton);
    expect(mockOnSelect).toHaveBeenCalledWith('joker');
  });


  it('is keyboard accessible', async () => {
    const user = userEvent.setup()
    render(
      <CharacterSelector
        characters={mockCharacters}
        selectedCharacterId="chandler"
        onCharacterSelect={mockOnSelect}
      />
    );
    const button = screen.getByRole('button', { name: /select chandler bing/i })
    await user.tab()
    expect(button).toHaveFocus()
  })
})
```

**Explanation:**

These tests validate the functionality of the `CharacterSelector` component:

*   **Rendering:** It checks that the component title and all character options are rendered correctly.
*   **Selection State:** It verifies that the selected character is visually distinct from the others and has the correct ARIA attributes.
*   **User Interaction:** It ensures that the `onCharacterSelect` callback is triggered when a user clicks on a character.
*   **Accessibility:** It confirms that the component is keyboard accessible, allowing users to navigate and select characters using the keyboard.

## 5. `ChatMessage` Component Tests

These tests are for the `ChatMessage` component, which is responsible for displaying individual messages in the chat.

### `components/chat/__tests__/ChatMessage.test.tsx`

```tsx
import React from 'react'
import { render, screen } from '@testing-library/react'
import '@testing-library/jest-dom'
import ChatMessage, { Message } from '../ChatMessage'

describe('ChatMessage', () => {
  const userMessage: Message = {
    id: '1',
    text: 'Hello there!',
    sender: 'user'
  }

  const characterMessage: Message = {
    id: '2',
    text: 'Could I BE any more excited to chat?',
    sender: 'character',
    characterName: 'Chandler',
    avatarUrl: '/chandler.jpg'
  }

  it('renders a user message correctly', () => {
    render(<ChatMessage message={userMessage} />)
    
    // Check if message text is present
    expect(screen.getByText(userMessage.text)).toBeInTheDocument()
    
    // Check if it has the user message styling
    const messageContainer = screen.getByText(userMessage.text).closest('div')
    expect(messageContainer).toHaveClass('bg-blue-500', 'text-white')
    
    // User messages shouldn't show character name
    expect(screen.queryByText('Chandler')).not.toBeInTheDocument()
  })

  it('renders a character message correctly', () => {
    render(<ChatMessage message={characterMessage} />)
    
    // Check if message text and character name are present
    expect(screen.getByText(characterMessage.text)).toBeInTheDocument()
    expect(screen.getByText(characterMessage.characterName!)).toBeInTheDocument()
    
    // Check if it has the character message styling
    const messageContainer = screen.getByText(characterMessage.text).closest('div')
    expect(messageContainer).toHaveClass('bg-gray-300', 'dark:bg-neutral-700')
  })

  it('handles messages without character name', () => {
    const messageWithoutName: Message = {
      id: '3',
      text: 'Anonymous message',
      sender: 'character'
    }
    
    render(<ChatMessage message={messageWithoutName} />)
    
    // Message text should be present
    expect(screen.getByText(messageWithoutName.text)).toBeInTheDocument()
    
    // No character name should be shown
    expect(screen.queryByTestId('character-name')).not.toBeInTheDocument()
  })

  // Edge cases
  it('handles empty message text', () => {
    const emptyMessage: Message = {
      id: '4',
      text: '',
      sender: 'user'
    }
    
    render(<ChatMessage message={emptyMessage} />)
    const messageContainer = screen.getByTestId('message-container')
    expect(messageContainer).toBeInTheDocument()
    expect(messageContainer).toHaveClass('flex', 'mb-3')
  })

  it('handles very long messages without breaking layout', () => {
    const longMessage: Message = {
      id: '5',
      text: 'a'.repeat(1000), // Very long message
      sender: 'user'
    }
    
    render(<ChatMessage message={longMessage} />)
    const messageContainer = screen.getByText(longMessage.text).closest('div')
    expect(messageContainer).toHaveClass('break-words')
  })

  it('handles messages with special characters', () => {
    const specialCharsMessage: Message = {
      id: '6',
      text: '<script>alert("XSS")</script>',
      sender: 'user'
    }
    
    render(<ChatMessage message={specialCharsMessage} />)
    expect(screen.getByText(specialCharsMessage.text)).toBeInTheDocument()
    // Text should be rendered as-is, not executed as HTML
    expect(document.querySelector('script')).not.toBeInTheDocument()
  })

  it('handles messages with multiple lines', () => {
    const multilineMessage: Message = {
      id: '7',
      text: 'Line 1\nLine 2\nLine 3',
      sender: 'character',
      characterName: 'Chandler'
    }
    
    render(<ChatMessage message={multilineMessage} />)
    // Use a custom function to match text content with newlines
    const textElement = screen.getByText((content) => {
      return content.includes('Line 1') && 
             content.includes('Line 2') && 
             content.includes('Line 3')
    })
    expect(textElement).toBeInTheDocument()
    expect(screen.getByText('Chandler')).toBeInTheDocument()
  })

  it('applies responsive width classes correctly', () => {
    render(<ChatMessage message={userMessage} />)
    const messageContainer = screen.getByText(userMessage.text).closest('div')
    expect(messageContainer).toHaveClass('max-w-xs', 'sm:max-w-md', 'md:max-w-lg', 'lg:max-w-xl', 'xl:max-w-2xl')
  })
})
```

**Explanation:**

These tests ensure that the `ChatMessage` component renders correctly for different types of messages:

*   **User and Character Messages:** It verifies that messages from the user and the character are rendered with the correct styling and content.
*   **Edge Cases:** It handles edge cases such as empty messages, very long messages, and messages with special characters to ensure the component is robust and secure.
*   **Layout:** It confirms that the component's layout is responsive and handles multiline messages correctly. 