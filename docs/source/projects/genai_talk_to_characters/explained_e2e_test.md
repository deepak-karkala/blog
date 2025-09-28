# End-to-End (E2E) Tests Explained

This document explains the end-to-end (E2E) tests for the chat application. These tests are designed to simulate real user interactions in a browser, ensuring that the entire system—from the frontend UI to the backend API and the LLM service—works together as expected.

The tests use [Playwright](https://playwright.dev/) to automate browser actions and assertions.

## `api/tests/e2e/test_chat.py`

```python
"""End-to-end tests for the chat interface."""

import os
from playwright.sync_api import Page, expect


BASE_URL = os.getenv("PLAYWRIGHT_BASE_URL", "http://localhost:3000")

# Define relative path for traces based on the current file's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACE_OUTPUT_DIR = os.path.join(CURRENT_DIR, "traces")

# Ensure the trace directory exists
os.makedirs(TRACE_OUTPUT_DIR, exist_ok=True)


def test_chandler_sends_greeting(page: Page):
    """Test that Chandler's initial greeting appears when the page loads."""
    try:
        page.goto(BASE_URL)
        greeting_text = "Could I BE any more ready to chat?"
        
        # Updated locator: targets outer div with data-testid, 
        # then inner div with class and text
        greeting_locator_text_part = f':has-text("{greeting_text}")'
        greeting_locator = (
            f'div[data-testid="message-container"] '
            f'div.message-bubble-character{greeting_locator_text_part}'
        )

        greeting_message = page.locator('text="Could I BE any more ready to chat?"')
        expect(greeting_message).to_be_visible(timeout=15000)
    finally:
        page.screenshot(path="test-failure.png")


def test_send_message_and_receive_reply(page: Page):
    """Test sending a message and receiving a reply from Chandler."""
    trace_path = os.path.join(
        TRACE_OUTPUT_DIR, "send_message_trace.zip"
    )
    page.context.tracing.start(
        screenshots=True, snapshots=True, sources=True
    )
    try:
        page.goto(BASE_URL)
        
        message_input_locator = 'textarea[data-testid="chat-input"]'
        message_input = page.locator(message_input_locator)
        expect(message_input).to_be_visible()
        message_input.fill("Hey Chandler, how are you?")
        
        # Click send
        send_button = page.locator('button[data-testid="send-button"]')
        send_button.click()
        
        # Check if user message appears
        user_message_text = "Hey Chandler, how are you?"
        # More specific locator for the user message
        user_message_locator = (
            f'div[data-testid="message-container"] '
            f'div.message-bubble-user:has-text("{user_message_text}")'
        )
        user_message = page.locator(user_message_locator)
        expect(user_message).to_be_visible()
        
        # Wait for typing indicator to appear
        typing_indicator = page.locator('[data-testid="typing-indicator"]')
        expect(typing_indicator).to_be_visible(timeout=10000)
        
        # Wait for typing indicator to disappear (meaning response is complete)
        expect(typing_indicator).to_be_hidden(timeout=30000)  # 30s timeout
        
        # Verify a response appeared in a character message bubble
        character_messages = page.locator('.message-bubble-character')
        last_message = character_messages.last
        expect(last_message).to_be_visible()
        expect(last_message).not_to_be_empty()
    finally:
        page.context.tracing.stop(path=trace_path)


def _test_image_upload_and_response(page: Page):
    """Test uploading an image and receiving a response about it."""
    trace_path = os.path.join(
        TRACE_OUTPUT_DIR, "image_upload_trace.zip"
    )
    page.context.tracing.start(
        screenshots=True, snapshots=True, sources=True
    )
    try:
        page.goto(BASE_URL)
        
        upload_button = page.locator('button[data-testid="attach-file-button"]')
        expect(upload_button).to_be_visible()

        # The file input is hidden, so we can't click it.
        # Instead, we'll listen for the file chooser and set the files.
        with page.expect_file_chooser() as fc_info:
            upload_button.click()
        file_chooser = fc_info.value
        file_chooser.set_files('tests/e2e/fixtures/test_image.jpg')

        # Verify image preview appears in chat IMMEDIATELY after upload
        uploaded_image = page.locator('[data-testid="image-preview"]')
        # Increased timeout for image to appear
        expect(uploaded_image).to_be_visible(timeout=10000)
        
        message_input_locator = 'textarea[data-testid="chat-input"]'
        message_input = page.locator(message_input_locator)
        message_input.fill("What do you think about this image?")
        
        send_button = page.locator('button[data-testid="send-button"]')
        send_button.click()
        
        # Wait for and verify response
        typing_indicator = page.locator('[data-testid="typing-indicator"]')
        expect(typing_indicator).to_be_visible()
        expect(typing_indicator).to_be_hidden(timeout=30000)
        
        # Verify response contains reference to the image
        character_messages = page.locator('.message-bubble-character')
        last_message = character_messages.last
        expect(last_message).to_be_visible()
        expect(last_message).not_to_be_empty()
    finally:
        page.context.tracing.stop(path=trace_path)
```

### Explanation

This file contains E2E tests that validate the chat functionality from a user's perspective. It uses Playwright to launch a browser, navigate to the application, and interact with the UI just as a real user would.

#### Key Features

*   **Playwright `Page` Fixture**: `pytest-playwright` automatically provides the `page` fixture, which is an instance of the Playwright `Page` object used to interact with the browser.
*   **Locators**: The tests use specific and robust locators (e.g., `[data-testid="chat-input"]`) to find elements on the page. Using `data-testid` attributes is a best practice as it decouples the tests from CSS classes or other implementation details that might change.
*   **Tracing**: For complex tests like `test_send_message_and_receive_reply`, Playwright's tracing is enabled. This feature records a detailed trace of the test execution, including screenshots, network requests, and a DOM snapshot for each action. If a test fails, this trace can be reviewed to quickly diagnose the problem.
*   **Screenshots**: A screenshot is taken on failure to provide a quick visual of what went wrong.

#### Test Cases

1.  **`test_chandler_sends_greeting`**:
    *   **Purpose**: To verify that the application initializes correctly and that the default character (Chandler) sends their initial greeting message when a user first visits the page.
    *   **Process**: It navigates to the application's base URL and waits for the specific greeting text to become visible on the screen. This simple test confirms that the frontend is rendering and the initial state from the backend/hook is being populated correctly.

2.  **`test_send_message_and_receive_reply`**:
    *   **Purpose**: This is the core E2E test, validating the entire chat message lifecycle.
    *   **Process**:
        1.  It finds the message input field, fills it with text, and clicks the send button.
        2.  It asserts that the user's sent message appears in the chat history.
        3.  It waits for the "typing indicator" to appear, which confirms that the backend has received the request and is processing it.
        4.  It then waits for the typing indicator to disappear, which signals that the streaming response from the LLM is complete.
        5.  Finally, it asserts that a new message bubble from the character has appeared and is not empty.

3.  **`_test_image_upload_and_response`**:
    *   **Purpose**: To test the multimodal capability of the chat, allowing users to upload an image and get a response about it.
    *   **Note**: This test is prefixed with an underscore (`_`), which means `pytest` will skip it by default. This is a common practice for tests that are a work-in-progress or require a specific setup.
    *   **Process**:
        1.  It simulates a user clicking the file attachment button.
        2.  It uses Playwright's `expect_file_chooser` to handle the native file selection dialog and "uploads" a fixture image.
        3.  It verifies that a preview of the uploaded image appears in the chat interface.
        4.  It then sends a text message related to the image and waits for a response, confirming the backend can handle image context. 