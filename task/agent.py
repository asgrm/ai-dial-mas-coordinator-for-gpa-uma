import json
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage
from pydantic import StrictStr

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:

    def __init__(self, endpoint: str, deployment_name: str, ums_agent_endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.ums_agent_endpoint = ums_agent_endpoint

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        #TODO:
        # 1. Create AsyncDial client (api_version='2025-01-01-preview')
        client: AsyncDial = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version="2025-01-01-preview"
        )

        # 2. Open stage for Coordination Request (StageProcessor will help with that)
        stage = StageProcessor.open_stage(choice, "Coordination Request")
        # 3. Prepare coordination request
        prepared_request = await self.__prepare_coordination_request(client, request)
        prepared_request_json = prepared_request.model_dump_json(indent=2)
        logger.info(f"\n Prepared coordinated request: {prepared_request_json}")
        # 4. Add to the stage generated coordination request and close the stage
        stage.append_content(f"```json\n\r{prepared_request_json}\n\r```\n\r")
        StageProcessor.close_stage_safely(stage)
        # 5. Handle coordination request (don't forget that all the content that will write called agent need to provide to stage)
        processing_stage = StageProcessor.open_stage(choice, f"Call {prepared_request.agent_name} Agent")
        agent_message = await self.__handle_coordination_request(
            prepared_request,
            choice,
            processing_stage,
            request,
        )
        logger.info(f"Agent response: {agent_message.json()}")
        StageProcessor.close_stage_safely(processing_stage)
        # 6. Generate final response based on the message from called agent
        final_respose = await self.__final_response(
            client,
            choice,
            request,
            agent_message
        )
        logger.info(f"Final response: {final_respose.json()}")
        return final_respose

    async def __prepare_coordination_request(self, client: AsyncDial, request: Request) -> CoordinationRequest:
        #TODO:
        # 1. Make call to LLM with prepared messages and COORDINATION_REQUEST_SYSTEM_PROMPT. For GPT model we can use
        #    `response_format` https://platform.openai.com/docs/guides/structured-outputs?example=structured-data and
        #    response will be returned in JSON format. The `response_format` parameter must be provided as extra_body dict
        #    {response_format": {"type": "json_schema","json_schema": {"name": "response","schema": CoordinationRequest.model_json_schema()}}}
        messages = self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT)
        response = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            # stream=False,
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": CoordinationRequest.model_json_schema()
                    }
                },
            }
        )
        # 2. Get content from response -> choice -> message -> content
        content = response.choices[0].message.content

        # 3. Load as dict
        dict_content = json.loads(content)
        # 4. Create CoordinationRequest from result, since CoordinationRequest is pydentic model, you can use `model_validate` method
        return CoordinationRequest.model_validate(dict_content)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        #TODO:
        # 1. Create array with messages, first message is system prompt and it is dict
        messages = [
            {
                "content": system_prompt,
                "role": Role.SYSTEM,
            }
        ]
        # 2. Iterate through messages from request and:
        #       - if user message that it has custom content and then add dict with user message and content (custom_content should be skipped)
        #       - otherwise append it as dict with excluded none fields (use `dict` method, despite it is deprecated since
        #         DIAL is using pydentic.v1)
        for m in request.messages:
            if m.custom_content and m.role == Role.USER:
                copied = deepcopy(m)
                messages.append(
                    {
                        "role": Role.USER,
                        "content": StrictStr(copied.content),
                    }
                )
            else:
                    messages.append(m.dict(exclude_none=True))

        return messages

    async def __handle_coordination_request(
            self,
            coordination_request: CoordinationRequest,
            choice: Choice,
            stage: Stage,
            request: Request
    ) -> Message:
        #TODO:
        # Make appropriate coordination requests to to proper agents and return the result
        if coordination_request.agent_name is AgentName.GPA:
            agent = GPAGateway(endpoint=self.endpoint)
        elif coordination_request.agent_name is AgentName.UMS:
            agent =  UMSAgentGateway(ums_agent_endpoint=self.ums_agent_endpoint)
        else:
            raise ValueError("Unknown Agent")

        return await agent.response(
            choice=choice,
            request=request,
            stage=stage,
            additional_instructions=coordination_request.additional_instructions,
        )

    async def __final_response(
            self,
            client: AsyncDial,
            choice: Choice,
            request: Request,
            agent_message: Message
    ) -> Message:
        #TODO:
        # 1. Prepare messages with FINAL_RESPONSE_SYSTEM_PROMPT
        messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)
        # 2. Make augmentation of retrieved agent response (as context) with user request (as user request)
        updated_user_request = f"## CONTEXT:\n {agent_message.content}\n ---\n ## USER_REQUEST: \n {messages[-1]['content']}"
        # 3. Update last message content with augmented prompt
        messages[-1]["content"] = updated_user_request
        # 4. Call LLM with streaming
        response = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            stream=True
        )
        # 5. Stream final response to choice
        content = ''
        async for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    choice.append_content(delta.content)
                    content += delta.content

        return Message(
            role=Role.ASSISTANT,
            content=StrictStr(content),
            custom_content=agent_message.custom_content
        )
