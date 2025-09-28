# Load Testing using Locust

```python
import random
from locust import HttpUser, task, between

class ChatUser(HttpUser):
    wait_time = between(1, 5)  # Simulate users waiting 1-5 seconds between tasks

    def on_start(self):
        """
        This method is called when a new user is started.
        We can use it to pre-load any data or set up the user.
        """
        self.questions = [
            "Could you BE any more helpful?",
            "Tell me a joke about commitment.",
            "What's the best way to handle a Monday?",
            "Explain the rules of 'Cups'.",
            "What is the meaning of 'Unagi'?",
        ]

    @task
    def send_chat_message(self):
        """
        This task simulates a user sending a chat message.
        """
        random_question = random.choice(self.questions)
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        json_payload = {
            "messages": [
                {
                    "role": "user",
                    "content": random_question
                }
            ]
        }

        self.client.post(
            "/api/v1/chat",
            json=json_payload,
            headers=headers,
            name="/api/v1/chat [stream]" # Name for reporting in Locust UI
        )
```

**Load Testing Results**
![Load Testing results](../../../../_static/projects/genai_talk_to_characters/load_testing.png)