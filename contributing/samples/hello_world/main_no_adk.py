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
from google.adk.models import GemmaModel  # Updated import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
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
    
    # Initialize Gemma model directly
    try:
        gemma_model = GemmaModel()
        logger.info("Gemma model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemma model: {str(e)}")
        return

    async def run_prompt(prompt: str):
        logger.info(f'User input: {prompt}')
        logger.info('Using Gemma 12B model for generation...')
        try:
            response = gemma_model.generate(prompt)
            logger.info(f'Gemma response: {response}')
            return response
        except Exception as e:
            logger.error(f'Error generating response: {str(e)}')
            return f"Error: {str(e)}"

    start_time = time.time()
    logger.info('Start time: %s', start_time)
    logger.info('------------------------------------')
    
    # Test the model with various prompts
    await run_prompt('Hi there!')
    await run_prompt('What is the capital of France?')
    await run_prompt('Can you help me write a simple Python function?')
    await run_prompt('Tell me a short joke.')
    
    end_time = time.time()
    logger.info('------------------------------------')
    logger.info('End time: %s', end_time)
    logger.info('Total time: %s seconds', end_time - start_time)


if __name__ == '__main__':
    asyncio.run(main())
