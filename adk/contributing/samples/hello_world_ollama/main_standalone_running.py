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
import warnings

import agent
from dotenv import load_dotenv
from google.adk.runners import InMemoryRunner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.cli.utils import logs
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.genai import types

load_dotenv(override=True)
warnings.filterwarnings('ignore', category=UserWarning)
logs.log_to_tmp_folder()


async def main():
  print("=" * 60)
  print("ADK Hello World Ollama - Command Line Interface")
  print("=" * 60)
  
  app_name = 'my_app'
  user_id_1 = 'user1'
  
  # Use InMemoryRunner which handles all services internally
  runner = InMemoryRunner(
      agent=agent.root_agent,
      app_name=app_name,
  )
  
  # Get the session service from the runner
  session_service = runner._in_memory_session_service
  session_11 = await session_service.create_session(
      app_name=app_name, user_id=user_id_1
  )

  async def run_prompt(session: Session, new_message: str):
    content = types.Content(
        role='user', parts=[types.Part.from_text(text=new_message)]
    )
    print(f'🤖 User: {new_message}')
    print(f'🤖 Agent: ', end='', flush=True)
    
    response_text = ""
    async for event in runner.run_async(
        user_id=user_id_1,
        session_id=session.id,
        new_message=content,
    ):
      breakpoint()
      # Print both function calls and actual text responses
      if event.content.parts and event.content.parts[0].text:
        text = event.content.parts[0].text

        print(f'{text}', end='', flush=True)
        # Check if it's a function call (JSON format)
        # if text.strip().startswith('{"name":') and text.strip().endswith('}'):
        #   print(f"[Function Call: {text}]", end='', flush=True)
        # else:
        #   # This is actual readable response text
        #   response_text += text
        #   print(text, end='', flush=True)
    
    # If no readable text was produced, show a message
    if not response_text.strip():
        print("(Agent made function calls but provided no readable response)")
    print()  # New line after response

  start_time = time.time()
  print('🚀 Starting conversation...')
  print('=' * 60)
  
  # Run predefined prompts
  
  # Interactive mode
  print('=' * 60)
  print('🎯 Interactive mode - Type your messages (or "quit" to exit):')
  print('=' * 60)
  
  while True:
    try:
      user_input = input('You: ').strip()
      if user_input.lower() in ['quit', 'exit', 'bye']:
        print('👋 Goodbye!')
        break
      if user_input:
        await run_prompt(session_11, user_input)
    except KeyboardInterrupt:
      print('\n👋 Goodbye!')
      break
    except EOFError:
      print('\n👋 Goodbye!')
      break
  
  end_time = time.time()
  print('=' * 60)
  print(f'⏱️  Total time: {end_time - start_time:.2f} seconds')
  print('=' * 60)


if __name__ == '__main__':
  asyncio.run(main())
