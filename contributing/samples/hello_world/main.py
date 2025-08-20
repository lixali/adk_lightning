# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import time
import logging
import os
import datetime
from google.adk.agents.run_config import RunConfig
from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.adk.sessions.session import Session
from google.genai import types
from google.adk.models import GemmaModel  # Updated import
from google.adk.agents import LlmAgent

# Configure logging
log_filename = f"hello_world_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(log_filename)  # File output
    ]
)
logger = logging.getLogger(__name__)

print("=" * 50)
logger.info("Starting ADK Hello World Sample with Gemma 12B Model")
print("=" * 50)


async def main():
    # Check environment
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        logger.error("HUGGING_FACE_TOKEN not found in environment!")
        return
    
    logger.info("Environment checked - HUGGING_FACE_TOKEN found")
    logger.info("Initializing Gemma 12B model...")
    
    # Define a simple config object
    class SimpleConfig:
        def __init__(self):
            self.model_registry = self
            self.registered_models = {}
        
        def register_model(self, name, model):
            self.registered_models[name] = model
            logger.info(f"Registered model '{name}' in config")
    
    config = SimpleConfig()
    
    # Initialize Gemma model and register it
    try:
        model = GemmaModel()
        config.model_registry.register_model("gemma-12b", model)
        logger.info(f"Gemma model loaded successfully on device: {model.device}")
    except Exception as e:
        logger.error(f"Failed to initialize Gemma model: {str(e)}")
        return
    
    # Create agent with Gemma model
    logger.info("Creating Gemma-powered agent...")
    root_agent = LlmAgent(
        name="GemmaAgent",
        model="gemma-12b",  # Match the registered model name
        instruction="""You are a helpful AI assistant powered by Gemma 12B.
        Keep your responses clear and concise."""
    )
    logger.info("Gemma agent created with configuration")

    app_name = 'my_app'
    user_id_1 = 'user1'
    logger.info("Initializing runner with Gemma agent...")
    runner = InMemoryRunner(
        agent=root_agent,
        app_name=app_name,
        config=config  # Pass the config with registered model
    )
    logger.info("Runner initialized with Gemma model!")

    session_11 = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id_1
    )

    async def run_prompt(session: Session, new_message: str = None):
        if new_message is None:
            new_message = input("Enter your message: ")
        
        content = types.Content(
            role='user', parts=[types.Part.from_text(text=new_message)]
        )
        logger.info(f'User input: {new_message}')
        logger.info('Using Gemma 12B model for generation...')
        async for event in runner.run_async(
            user_id=user_id_1,
            session_id=session.id,
            new_message=content,
        ):
            if event.content.parts and event.content.parts[0].text:
                logger.info(f'Agent response: {event.content.parts[0].text}')

    async def run_prompt_bytes(session: Session, new_message: str):
        content = types.Content(
            role='user',
            parts=[
                types.Part.from_bytes(
                    data=str.encode(new_message), mime_type='text/plain'
                )
            ],
        )
        logger.info(f'User input (bytes): {new_message}')
        async for event in runner.run_async(
            user_id=user_id_1,
            session_id=session.id,
            new_message=content,
            run_config=RunConfig(save_input_blobs_as_artifacts=True),
        ):
            if event.content.parts and event.content.parts[0].text:
                logger.info(f'Agent response: {event.content.parts[0].text}')

    # Removed dice rolling functionality - focusing on conversation only

    start_time = time.time()
    logger.info('Start time: %s', start_time)
    logger.info('------------------------------------')
    logger.info('Gemma 12B agent is ready for conversation!')
    logger.info('Session created and waiting for real user input...')
    
    # Interactive conversation
    await run_prompt(session_11)  # This will prompt user for input
    
    end_time = time.time()
    logger.info('------------------------------------')
    logger.info('End time: %s', end_time)
    logger.info('Setup time: %s seconds', end_time - start_time)


if __name__ == '__main__':
    asyncio.run(main())
