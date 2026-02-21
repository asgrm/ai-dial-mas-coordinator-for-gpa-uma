import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import MASCoordinator
from task.logging_config import setup_logging, get_logger

DIAL_ENDPOINT = os.getenv('DIAL_ENDPOINT', "http://localhost:8080")
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
UMS_AGENT_ENDPOINT = os.getenv('UMS_AGENT_ENDPOINT', "http://localhost:8042")
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

setup_logging(log_level=LOG_LEVEL)
logger = get_logger(__name__)


class MASCoordinatorApplication(ChatCompletion):

    async def chat_completion(self, request: Request, response: Response) -> None:
        #TODO:
        # 1. Create single choice with context manager
        conversation_id = request.headers.get('x-conversation-id', 'unknown')
        logger.info(f"Processing chat completion request  for conversation_id {conversation_id}")
        try:
            with response.create_single_choice() as choice:
        # 2. Create MASCoordinator and handle request
                mas_coordinator = MASCoordinator(DIAL_ENDPOINT, DEPLOYMENT_NAME, UMS_AGENT_ENDPOINT)
                await mas_coordinator.handle_request(choice, request)
        except Exception as e:
            logger.error(
                f"Error processing chat completion for conversation_id {conversation_id}: {str(e)}",
                exc_info=True
            )
            raise e



#TODO:
# 1. Create DIALApp
# 2. Create MASCoordinatorApplication
# 3. Add to created DIALApp chat_completion with:
#       - deployment_name="mas-coordinator"
#       - impl=agent_app
# 4. Run it with uvicorn: `uvicorn.run({CREATED_DIAL_APP}, port=8055, host="0.0.0.0")`

app: DIALApp = DIALApp()
agent_app = MASCoordinatorApplication()
app.add_chat_completion(deployment_name="mas-coordinator", impl=agent_app)
logger.info("App initialized")

if __name__ == "__main__":
    import sys

    if 'pydevd' in sys.modules:
        logger.info("Running in debug mode")
        config = uvicorn.Config(app, port=8055, host="0.0.0.0", log_level="info")
        server = uvicorn.Server(config)
        import asyncio
        asyncio.run(server.serve())
    else:
        logger.info("Starting uvicorn server on 0.0.0.0:8055")
        uvicorn.run(app, port=8055, host="0.0.0.0", log_level="info")