# Plan: CI/CD

**Popular Frameworks/Tools for Web App Testing:**

*   **Frontend Unit/Integration Tests:**
    *   **Jest:** A popular JavaScript testing framework.
    *   **React Testing Library (RTL):** For testing React components in a way that resembles how users interact with them. Often used with Jest.
*   **Backend Unit/Integration Tests (Python):**
    *   **Pytest:** A mature and widely used Python testing framework known for its simplicity and powerful features.
*   **End-to-End (E2E) Tests:**
    *   **Playwright:** Excellent choice! It's modern, supports multiple browsers (Chromium, Firefox, WebKit), multiple languages (including Python), and has features like auto-waits, rich selectors, and tracing.
    *   **Cypress:** Another popular E2E testing framework, primarily JavaScript-based.
    *   **Selenium:** A long-standing tool, but Playwright is generally considered more modern and easier to use for many scenarios.

**Playwright Python API is indeed a very good option for your E2E tests.** It will allow you to write tests that interact with your deployed application or a preview environment just like a user would.

---

## 1. Factors to Consider While Planning CI/CD

1.  **Repository Structure:**
    *   **Monorepo:** Your current plan (frontend and backend in one repo) requires the CI/CD pipeline to handle both. Vercel can be configured to build from subdirectories.
2.  **Branching Strategy:**
    *   A simple strategy like **GitHub Flow** (create branches from `main`, open PRs to `main`, merge to `main` after review/checks) works well with Vercel's preview deployments.
3.  **Triggering Events:**
    *   **CI:** Run on every push to any branch (especially feature branches) and on every pull request to `main`.
    *   **CD:** Deploy to production from `main` branch. Vercel automatically creates preview deployments for PRs.
4.  **Testing Strategy (automated checks):**
    *   **Linting:** Enforce code style and catch syntax errors early (ESLint for frontend, Flake8/Black for backend).
    *   **Unit Tests:** Test individual functions/components in isolation (Jest/RTL for Next.js, Pytest for FastAPI).
    *   **Integration Tests:** Test interactions between components/modules.
    *   **E2E Tests (Playwright):** Test full user flows through the UI. These are crucial but can be slower, so consider when to run them (e.g., on PRs to `main` and before production deployment).
5.  **Build Process:**
    *   **Frontend (Next.js):** `next build`. Vercel handles this automatically.
    *   **Backend (FastAPI):** If deploying to Vercel Serverless, Vercel needs to install Python dependencies (`requirements.txt`).
6.  **Environment Variables & Secrets Management:**
    *   **Vercel:** Has its own system for managing environment variables for different environments (Production, Preview, Development).
    *   **GitHub Actions (for CI):** Use GitHub Secrets to store sensitive information needed during CI runs (e.g., API keys for test environments, if any).
    *   Never commit secrets directly to the repository.
7.  **Deployment Targets & Strategy:**
    *   **Frontend (Next.js):** Deployed by Vercel.
    *   **Backend (FastAPI):** Deployed as Vercel Serverless Functions. The FastAPI app in your `backend/` directory will be configured as the API source for Vercel.
    *   **RunPod LLM Endpoint:** This is a separate deployment. Your Vercel-deployed backend will *call* this RunPod endpoint. The CI/CD for the *application code* doesn't deploy the RunPod model but ensures the app can connect to it. You'll need the RunPod endpoint URL as an environment variable in Vercel.
8.  **Pipeline Speed & Efficiency:**
    *   Optimize test execution times.
    *   Use caching for dependencies in GitHub Actions.
9.  **Rollbacks:**
    *   Vercel provides immutable deployments, making rollbacks to previous versions straightforward.
10. **Notifications:**
    *   GitHub Actions can notify on success/failure (e.g., via Slack or email). Vercel also provides deployment notifications.
11. **Security Scanning (Optional but Recommended):**
    *   Tools like Snyk, Dependabot (built into GitHub) for dependency vulnerability scanning.
    *   Static Application Security Testing (SAST) tools.

---

## 2. Detailed Step-by-Step Implementation Plan for CI/CD

This plan uses GitHub Actions for CI and Vercel's native GitHub integration for CD.

**Phase 0: Prerequisites**

*   **Step 0.1:** Ensure your code is in a GitHub repository.
*   **Step 0.2:** Sign up for Vercel and link your GitHub account.
*   **Step 0.3:** Familiarize yourself with GitHub Actions workflow syntax (`.yml`).
*   **Step 0.4:** Ensure `frontend/package.json` has scripts for linting and testing.
*   **Step 0.5:** Ensure `backend/requirements.txt` is up-to-date. Add testing and linting tools to a `requirements-dev.txt`.

**Phase 1: CI Setup with GitHub Actions**

We'll create a GitHub Actions workflow file (e.g., `.github/workflows/ci.yml`).

*   **Step 1.1: Create Workflow File**
    *   In your project root, create `.github/workflows/ci.yml`.
    ```yaml
    name: Application CI

    on:
      push:
        branches:
          - main # Runs on pushes to main
          - '*'  # Runs on pushes to any branch
      pull_request:
        branches:
          - main # Runs on PRs targeting main

    jobs:
      lint:
        name: Lint Code
        runs-on: ubuntu-latest
        steps:
          - name: Checkout code
            uses: actions/checkout@v4

          - name: Setup Node.js for Frontend Linting
            uses: actions/setup-node@v4
            with:
              node-version: '20' # Or your project's Node.js version
              cache: 'npm'
              cache-dependency-path: frontend/package-lock.json

          - name: Install Frontend Dependencies & Lint
            working-directory: ./frontend
            run: |
              npm ci
              npm run lint # Assuming you have 'lint' script in package.json (e.g., "eslint .")

          - name: Setup Python for Backend Linting
            uses: actions/setup-python@v5
            with:
              python-version: '3.10' # Or your project's Python version

          - name: Install Backend Dependencies & Lint
            working-directory: ./backend
            run: |
              pip install -r requirements-dev.txt # Should contain flake8, black
              flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
              black . --check

      # Opional: Frontend Unit/Integration Tests (Jest/RTL)
      frontend-tests:
        name: Frontend Tests
        runs-on: ubuntu-latest
        needs: lint # Run after linting
        steps:
          - name: Checkout code
            uses: actions/checkout@v4
     
          - name: Setup Node.js
            uses: actions/setup-node@v4
            with:
              node-version: '20'
              cache: 'npm'
              cache-dependency-path: frontend/package-lock.json
     
          - name: Install Frontend Dependencies & Run Tests
            working-directory: ./frontend
            run: |
              npm ci
              npm test # Assuming 'test' script runs Jest

      #Optional: Backend Unit/Integration Tests (Pytest)
      backend-tests:
        name: Backend Tests
        runs-on: ubuntu-latest
        needs: lint # Run after linting
        steps:
          - name: Checkout code
            uses: actions/checkout@v4
     
          - name: Setup Python
            uses: actions/setup-python@v5
            with:
              python-version: '3.10'
     
          - name: Install Backend Dependencies & Run Tests
            working-directory: ./backend
            run: |
              pip install -r requirements-dev.txt # Should contain pytest
              pytest

     e2e-tests:
        name: End-to-End Tests (Playwright)
        runs-on: ubuntu-latest
        # needs: [frontend-tests, backend-tests] # If you have unit tests
        needs: lint # At least run after linting
        if: github.event_name == 'pull_request' || github.ref == 'refs/heads/main' # Run on PRs to main or direct pushes to main
        steps:
          - name: Checkout code
            uses: actions/checkout@v4

          - name: Setup Python for Playwright
            uses: actions/setup-python@v5
            with:
              python-version: '3.10'

          - name: Install Playwright Browsers
            run: python -m playwright install --with-deps # Installs browsers needed by Playwright

          - name: Install Backend Dependencies (for Playwright tests)
            # Playwright tests might need to interact with a running backend
            # For E2E against Vercel preview, this might not be needed here.
            # But if running backend locally for tests:
            working-directory: ./backend
            run: pip install -r requirements.txt -r requirements-dev.txt # Playwright itself

          - name: Run Playwright Tests
            working-directory: ./backend # Assuming Playwright tests are in backend/tests/e2e
            env:
              # If testing against Vercel preview URL:
              # VERCEL_PREVIEW_URL: ${{ github.event.deployment_status.target_url }} # This is complex to get reliably
              # Easier: Pass base URL as an argument or env var to Playwright tests.
              # For PRs, Vercel posts a comment with the preview URL. Tests could run against that.
              # For now, let's assume tests might run against a locally started dev server or a fixed staging URL
              # For CI, it's best if tests run against the actual Vercel preview URL.
              # This requires Vercel deployment to finish first. Vercel GitHub App handles this.
              # For simplicity, let's assume Playwright tests are configured to target the appropriate URL.
              # Example if Playwright tests are configured to run against a fixed URL set in GitHub Secrets:
              # BASE_URL: ${{ secrets.TEST_BASE_URL }}
              # For tests against Vercel preview, often done via Vercel's "Checks" integration.
              # Let's assume Playwright tests are within the backend structure
              PLAYWRIGHT_BASE_URL: ${{ secrets.VERCEL_PREVIEW_URL || 'http://localhost:3000' }} # Fallback for local
            run: |
              # If you need to start the frontend dev server for Playwright:
              # (cd ../frontend && npm ci && npm run dev &)
              # If you need to start the backend dev server for Playwright:
              # (python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &)
              # sleep 30 # Wait for servers to start
              
              # Recommended for Playwright in CI: Run against a Vercel preview URL
              # This means this job might need to be triggered *after* Vercel deployment check is successful
              # Or, Vercel itself can trigger these tests via its GitHub Checks integration
              # For now, this assumes Playwright is set up to run against a known URL
              pytest tests/e2e # Assuming tests are in backend/tests/e2e and run with pytest
              # Or directly: python -m playwright test tests/e2e
    ```
*   **Step 1.2: Add Linting Scripts to `frontend/package.json`**
    ```json
    // frontend/package.json
    "scripts": {
      // ... other scripts
      "lint": "eslint . --ext .js,.jsx,.ts,.tsx"
    }
    ```
*   **Step 1.3: Create `backend/requirements-dev.txt`**
    ```
    # backend/requirements-dev.txt
    pytest
    flake8
    black
    playwright
    # Add any other testing specific dependencies
    ```
*   **Step 1.4: Write Playwright E2E Tests (Example)**
    *   Install Playwright: `pip install playwright` and `python -m playwright install` locally.
    *   Create tests, e.g., in `backend/tests/e2e/test_chat.py`:
    ```python
    # backend/tests/e2e/test_chat.py
    import pytest
    import os
    from playwright.sync_api import Page, expect

    BASE_URL = os.getenv("PLAYWRIGHT_BASE_URL", "http://localhost:3000") # Default to local dev

    def test_chandler_sends_greeting(page: Page):
        page.goto(BASE_URL)
        # Wait for Chandler's initial greeting to appear (adjust selector and text)
        greeting_message = page.locator('div:has-text("Could I BE any more ready to chat?")').nth(-1) # Simplistic selector
        expect(greeting_message).to_be_visible(timeout=15000) # Increased timeout for initial load + LLM call

    def test_send_message_and_receive_reply(page: Page):
        page.goto(BASE_URL)
        
        # Ensure character (Chandler) is selected or select him (if multiple chars UI exists)
        # For MVP, Chandler is default.
        
        # Type a message
        message_input = page.locator('textarea[placeholder*="Type your message"]') # Adjust selector
        expect(message_input).to_be_visible()
        message_input.fill("Hey Chandler, how are you?")
        
        # Click send
        send_button = page.locator('button:has-text("Send")') # Adjust selector
        send_button.click()
        
        # Check if user message appears
        user_message = page.locator('div:has-text("Hey Chandler, how are you?")').nth(-1)
        expect(user_message).to_be_visible()
        
        # Wait for and check character's reply (this will be LLM dependent and might need longer timeouts)
        # This is a placeholder for a more robust way to identify the character's reply
        # A more robust selector would target the last message bubble from the character
        # This test is very basic and will need refinement.
        reply_locator_base = 'div[class*="message-bubble-character"]' # Fictional class
        
        # Wait for a new message from the character
        # This is tricky because LLM responses are non-deterministic.
        # You might look for the "typing" indicator to disappear, then for a new message.
        # For simplicity, let's just wait for any text.
        page.wait_for_timeout(10000) # Give LLM time to respond, very crude
        
        # Verify some text appears in a character reply bubble
        # This is a very weak assertion for LLM testing
        # More advanced: check for style, keywords, or use snapshot testing if applicable.
        all_messages = page.locator(f"{reply_locator_base} p") # Assuming text is in a <p>
        expect(all_messages.last).not_to_be_empty(timeout=20000) # Check last character message isn't empty
    ```
    *   **Note on Playwright E2E tests for LLM apps:** Testing LLM outputs is hard due to non-determinism. Focus E2E tests on:
        *   UI rendering correctly.
        *   Messages being sent and user messages appearing.
        *   *A* response appearing (without being too strict on the exact wording initially).
        *   Presence of key UI elements (typing indicator, character avatar).
    *   The `VERCEL_PREVIEW_URL` is tricky to get directly into the GitHub Actions Playwright job *before* Playwright runs *unless* Vercel triggers the check. A common pattern is for Vercel to run the tests against its own preview deployment using GitHub Checks API.
    *   For simpler CI initially, you might run Playwright against a locally spun-up dev environment in CI, or have a dedicated staging environment.

**Phase 2: CD Setup with Vercel for GitHub**

*   **Step 2.1: Import Project to Vercel**
    *   Go to your Vercel dashboard.
    *   Click "Add New..." -> "Project".
    *   Select your GitHub repository. Vercel will automatically detect it's a Next.js project.
*   **Step 2.2: Configure Project Settings on Vercel**
    *   **Root Directory:**
        *   If Vercel correctly detects `frontend` as the root of the Next.js app, great.
        *   If not, set "Root Directory" to `frontend`.
    *   **Build & Output Settings:** Vercel usually auto-detects these for Next.js.
    *   **Serverless Functions for Backend:**
        *   Vercel can deploy Python serverless functions. Your FastAPI app needs to be structured so Vercel can find it.
        *   Create a `vercel.json` file in the **root of your GitHub repository** (not `frontend/` or `backend/`):
        ```json
        // vercel.json (in project root)
        {
          "version": 2,
          "builds": [
            {
              "src": "frontend/next.config.mjs", // Or package.json if that's your entry point indicator
              "use": "@vercel/next"
            },
            {
              "src": "backend/app/main.py", // Path to your FastAPI app instance
              "use": "@vercel/python"
            }
          ],
          "routes": [
            {
              "src": "/api/(.*)", // All requests to /api/...
              "dest": "/backend/app/main.py" // Route them to your FastAPI app
            },
            {
              "src": "/(.*)", // All other requests
              "dest": "/frontend/$1" // Route them to the Next.js app
            }
          ]
        }
        ```
        *   Adjust `src` paths in `vercel.json` carefully based on your directory structure and where your FastAPI app object is defined in `backend/app/main.py`.
        *   Ensure your `backend/app/main.py` has an `app` object that Vercel can serve (e.g., `app = FastAPI()`).
*   **Step 2.3: Configure Environment Variables on Vercel**
    *   In your Vercel project settings, go to "Environment Variables".
    *   Add all necessary variables for Production, Preview, and Development environments:
        *   `GEMINI_API_KEY`
        *   `LANGSMITH_API_KEY`
        *   `LANGCHAIN_API_KEY` (if distinct from LANGSMITH_API_KEY)
        *   `LANGCHAIN_TRACING_V2="true"`
        *   `LANGCHAIN_PROJECT="Your LangSmith Project Name"`
        *   `LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"`
        *   `RUNPOD_MODEL_ENDPOINT_URL` (URL for your deployed fine-tuned model on RunPod)
        *   `LLM_MODE="GEMINI"` or `LLM_MODE="FINETUNED"` (for your backend to switch LLM services)
*   **Step 2.4: Deployment**
    *   Once configured, Vercel will automatically deploy your `main` branch to production.
    *   When you open a Pull Request against `main`, Vercel will create a Preview Deployment with a unique URL.
    *   The GitHub Actions CI workflow (defined in `ci.yml`) will run on these PRs and pushes.
*   **Step 2.5: (Optional) Vercel GitHub Checks for E2E Tests**
    *   Instead of running Playwright in your main `ci.yml` targeting a hardcoded URL, you can configure Vercel to trigger a GitHub Actions workflow *after* its preview deployment is ready. This workflow would then run Playwright tests against the specific Vercel preview URL. This is more advanced to set up but provides the most accurate E2E testing.
    *   Vercel can also integrate with third-party E2E testing services that pick up preview URLs.

**Phase 3: Iteration and Refinement**

*   **Step 3.1: Monitor CI/CD Pipelines:** Check GitHub Actions logs and Vercel deployment logs for any issues.
*   **Step 3.2: Refine Tests:** Continuously improve your tests. Add more coverage. Make E2E tests more robust for LLM interactions (this is an ongoing challenge).
*   **Step 3.3: Optimize Pipeline Speed:** Look for bottlenecks. Use caching effectively.
*   **Step 3.4: Secure Secrets:** Regularly review how secrets are handled.
