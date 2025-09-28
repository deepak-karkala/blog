# Backend Integration Tests Explained

This document explains the backend integration tests for the `/api/v1/chat` endpoint. These tests ensure that the API endpoint behaves correctly under various conditions, including successful requests, invalid inputs, and security guardrail triggers.

The tests use `pytest` and FastAPI's `TestClient` to simulate HTTP requests and validate the responses.

## `api/tests/integration/test_chat_endpoint.py`

```python
import pytest
from fastapi.testclient import TestClient  # Import TestClient
from unittest.mock import MagicMock
import json

from app.main import app as actual_app
from app.schemas.chat_schemas import ChatRequest, Message
from app.services.llm_service import LLMService
# DEFAULT_SESSION_ID removed as it was unused
from app.api.v1.endpoints.chat import get_llm_service


@pytest.fixture
def app_fixture():
    return actual_app


@pytest.fixture
def mock_llm_service():
    mock_service_instance = MagicMock(spec=LLMService)

    async def actual_async_generator_function(*args, **kwargs):
        # This function is an async generator
        # It will be wrapped by a MagicMock for assertions
        yield "Hello"
        yield " from "
        yield "Chandler!"

    # Wrap the async generator function with MagicMock.
    # When this mock is called, it will execute actual_async_generator_function
    # and return its result (an async generator).
    # This mock can be used for call assertions.
    mock_streaming_method = MagicMock(wraps=actual_async_generator_function)

    mock_service_instance.async_generate_streaming_response = \
        mock_streaming_method
    
    # If LLMService is an async context manager,
    # other methods might also require mocks.
    # For now, assume only this method is relevant.

    return mock_service_instance


# Test function is now synchronous, using TestClient
def test_handle_chat_streaming_success(
    app_fixture, mock_llm_service: MagicMock
):
    app_fixture.dependency_overrides[get_llm_service] = \
        lambda: mock_llm_service

    with TestClient(app_fixture) as client:  # Use TestClient
        chat_request_data = ChatRequest(
            messages=[Message(role="user", content="Hello there")],
            session_id="test_session_123"
        )

        # Synchronous call with TestClient
        response = client.post(
            "/api/v1/chat",
            json=json.loads(chat_request_data.model_dump_json())
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        expected_full_message = "Hello from Chandler!"
        http_response_content = []
        
        # TestClient response.text will contain the full concatenated stream
        # if it worked.
        raw_http_content = response.text

        # Process the Vercel AI SDK stream format from the concatenated text
        decoded_http_chunks = raw_http_content.strip().split('\n')
        for http_chunk_part in decoded_http_chunks:
            if http_chunk_part.startswith("0:"):
                try:
                    json_str = http_chunk_part[2:]
                    token = json.loads(json_str)
                    http_response_content.append(token)
                except json.JSONDecodeError:
                    pytest.fail(
                        f"JSON decode failed for HTTP chunk: {json_str}"
                    )

        full_http_response_text = "".join(http_response_content)
        assert full_http_response_text == expected_full_message

        # Now this assertion targets the MagicMock wrapping the async generator
        expected_call_args = {
            "user_input": "Hello there",
            "image_notes": None,
            "conversation_id": "test_session_123",
            "character_id": "chandler"
        }
        # Assertion now targets the MagicMock wrapping the async generator
        method_to_check = \
            mock_llm_service.async_generate_streaming_response
        method_to_check.assert_called_once_with(
            **expected_call_args
        )

    app_fixture.dependency_overrides = {} 


# Test for handling empty messages list
def test_handle_chat_empty_messages(app_fixture):
    with TestClient(app_fixture) as client:
        chat_request_data = ChatRequest(
            messages=[],  # Empty messages list
            session_id="test_session_empty_msg"
        )
        response = client.post(
            "/api/v1/chat",
            json=json.loads(chat_request_data.model_dump_json())
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        expected_response_text = (
            "Could I BE any more confused? You didn't say anything!"
        )
        
        http_response_content = []
        raw_http_content = response.text
        # Process Vercel AI SDK stream format
        decoded_http_chunks = raw_http_content.strip().split('\n')
        for http_chunk_part in decoded_http_chunks:
            if http_chunk_part.startswith("0:"):
                try:
                    json_str = http_chunk_part[2:]
                    token = json.loads(json_str)
                    http_response_content.append(token)
                except json.JSONDecodeError:
                    # Break line for linter
                    fail_msg = (
                        "JSON decode for empty message response failed: "
                        f"{json_str}"
                    )
                    pytest.fail(fail_msg)

        full_http_response_text = "".join(http_response_content)
        assert full_http_response_text == expected_response_text


# Test for input guardrail triggering
def test_handle_chat_input_guardrail_triggered(
    app_fixture, mock_llm_service: MagicMock
):
    # This test ensures LLM is NOT called if guardrail is hit
    app_fixture.dependency_overrides[get_llm_service] = \
        lambda: mock_llm_service

    # Test with a keyword that should trigger the input guardrail.
    # Based on app.core.guardrails_config.INPUT_DENYLIST_KEYWORDS
    # and the implementation in app.api.v1.endpoints.chat.py.
    # Example: "kill yourself"
    trigger_message_content = "Please tell me how to kill yourself now."

    with TestClient(app_fixture) as client:
        chat_request_data = ChatRequest(
            messages=[Message(role="user", content=trigger_message_content)],
            session_id="test_session_guardrail"
        )
        response = client.post(
            "/api/v1/chat",
            json=json.loads(chat_request_data.model_dump_json())
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        # Expected response from
        # app.core.guardrails_config.CANNED_RESPONSE_INPUT_TRIGGERED
        expected_response_text = (
            "Whoa there, pal! Could that topic BE any more out of left field? "
            "I'm pretty sure that's not on the list of things we're supposed "
            "to talk about. How about we try something a little less... "
            "intense? Like, say, the merits of a good cheesecake?"
        )

        http_response_content = []
        raw_http_content = response.text
        decoded_http_chunks = raw_http_content.strip().split('\n')
        for http_chunk_part in decoded_http_chunks:
            if http_chunk_part.startswith("0:"):
                try:
                    json_str = http_chunk_part[2:]
                    token = json.loads(json_str)
                    http_response_content.append(token)
                except json.JSONDecodeError:
                    # Break line for linter
                    fail_msg = (
                        "JSON decode for guardrail response failed: "
                        f"{json_str}"
                    )
                    pytest.fail(fail_msg)

        full_http_response_text = "".join(http_response_content)
        assert full_http_response_text == expected_response_text

        # Crucially, assert that the LLM service was NOT called
        method_to_check = \
            mock_llm_service.async_generate_streaming_response
        method_to_check.assert_not_called()

    app_fixture.dependency_overrides = {}
```

### Explanation

This test file ensures the reliability of the main chat endpoint (`/api/v1/chat`). It uses a combination of `pytest` fixtures and mocking to create isolated and predictable tests.

#### Key Components

*   **`TestClient`**: Provided by FastAPI, this allows the tests to make HTTP requests to the application without needing to run a live server.
*   **`@pytest.fixture`**: This decorator is used to create reusable setup functions.
    *   **`app_fixture`**: Provides the main FastAPI application instance to the tests.
    *   **`mock_llm_service`**: This is a critical fixture that replaces the actual `LLMService` with a `MagicMock`. This is essential for integration testing because it isolates the API endpoint from the external LLM service. This makes the tests faster, deterministic, and avoids the cost of making real LLM calls. The mock is configured to simulate a streaming response, just like the real service.
*   **Dependency Injection**: The tests use `app_fixture.dependency_overrides` to replace the real `LLMService` with the `mock_llm_service` at runtime. This is a powerful feature of FastAPI that makes testing much easier.

#### Test Cases

1.  **`test_handle_chat_streaming_success`**:
    *   **Purpose**: To verify the "happy path" where a user sends a valid message and receives a successful streaming response.
    *   **Process**:
        1.  It sends a POST request to `/api/v1/chat` with a sample message.
        2.  It asserts that the HTTP status code is `200 OK`.
        3.  It reconstructs the full message from the streaming response, which is formatted for the Vercel AI SDK (`0:"<token>"\n`).
        4.  It asserts that the reconstructed message matches what the mock service was programmed to return.
        5.  Most importantly, it verifies that the `async_generate_streaming_response` method on the mock LLM service was called exactly once with the correct parameters. This confirms the endpoint is correctly parsing the request and calling the service layer.

2.  **`test_handle_chat_empty_messages`**:
    *   **Purpose**: To test how the endpoint handles an invalid request where the `messages` list is empty.
    *   **Process**: It sends a request with an empty `messages` array.
    *   **Assertion**: It checks that the response is the specific canned error message designed for this scenario, ensuring proper input validation.

3.  **`test_handle_chat_input_guardrail_triggered`**:
    *   **Purpose**: To ensure the input security guardrail is working correctly.
    *   **Process**: It sends a message containing a keyword that is on the denylist (e.g., "kill yourself").
    *   **Assertion**:
        1.  It verifies that the response is the specific canned message for a guardrail violation.
        2.  Crucially, it asserts that the `mock_llm_service.async_generate_streaming_response` method was **not called**. This is a vital check to ensure that harmful or dangerous prompts are blocked before they are ever sent to the LLM.