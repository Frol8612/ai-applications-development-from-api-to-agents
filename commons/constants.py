"""
Configuration constants for AI service integrations.

This module centralizes all API endpoints, API keys, and default configuration
values used across different AI service providers (OpenAI, Anthropic, Gemini).

All API keys are loaded from environment variables for security.
"""
from dotenv import load_dotenv
import os
load_dotenv()

API_PATH = os.getenv("DIAL_PATH")
API_VERSION = os.getenv("DIAL_API_VERSION")

# Default system prompt used across all AI services
DEFAULT_SYSTEM_PROMPT = "You are an assistant who answers concisely and informatively."

# OpenAI API configuration
# OPENAI_HOST = "https://api.openai.com"
# OPENAI_CHAT_COMPLETIONS_ENDPOINT = f"{OPENAI_HOST}/v1/chat/completions"
OPENAI_HOST = "https://ai-proxy.lab.epam.com"
# OPENAI_CHAT_COMPLETIONS_ENDPOINT = f"{OPENAI_HOST}/openai/deployments/gpt-4.1-mini-2025-04-14"
OPENAI_CHAT_COMPLETIONS_ENDPOINT = (
    f"{OPENAI_HOST}/openai/deployments/"
    "gpt-4.1-mini-2025-04-14/chat/completions"
)
OPENAI_RESPONSES_ENDPOINT = f"{OPENAI_HOST}/openai/deployments/gpt-4.1-mini-2025-04-14"
OPENAI_EMBEDDINGS_ENDPOINT = f"{OPENAI_HOST}/v1/embeddings"
OPENAI_API_KEY = os.getenv('DIAL_KEY', '')

# Anthropic API configuration
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# Google Gemini API configuration
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# User Service API configuration
USER_SERVICE_ENDPOINT = "http://localhost:8041"
