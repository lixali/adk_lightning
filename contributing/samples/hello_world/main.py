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
        instruction="""You are a helpful AI assistant powered by Gemma 12B. You can:
        1. Roll dice and keep track of results
        2. Respond to general questions
        3. Process both text and byte inputs
        Keep your responses clear and concise."""
    )
    logger.info("Gemma agent created with configuration")

    app_name = 'my_app'
    user_id_1 = 'user1'
    logger.info("Initializing runner with Gemma agent...")
    runner = InMemoryRunner(
        agent=root_agent,
        app_name=app_name
    )
    logger.info("Runner initialized with Gemma model!")

    session_11 = await runner.session_service.create_session(
        app_name=app_name, user_id=user_id_1
    )

    async def run_prompt(session: Session, new_message: str):
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

    async def check_rolls_in_state(rolls_size: int):
        session = await runner.session_service.get_session(
            app_name=app_name, user_id=user_id_1, session_id=session_11.id
        )
        assert len(session.state['rolls']) == rolls_size
        for roll in session.state['rolls']:
            assert roll > 0 and roll <= 100

    start_time = time.time()
    logger.info('Start time: %s', start_time)
    logger.info('------------------------------------')
    await run_prompt(session_11, 'Hi')
    await run_prompt(session_11, 'Roll a die with 100 sides')
    await check_rolls_in_state(1)
    await run_prompt(session_11, 'Roll a die again with 100 sides.')
    await check_rolls_in_state(2)
    await run_prompt(session_11, 'What numbers did I got?')
    await run_prompt_bytes(session_11, 'Hi bytes')
    logger.info(
        "Artifact keys: %s",
        await runner.artifact_service.list_artifact_keys(
            app_name=app_name, user_id=user_id_1, session_id=session_11.id
        )
    )
    end_time = time.time()
    logger.info('------------------------------------')
    logger.info('End time: %s', end_time)
    logger.info('Total time: %s seconds', end_time - start_time)


if __name__ == '__main__':
    asyncio.run(main())
