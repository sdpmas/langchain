<![CDATA[
"""Wrapper around MosaicML APIs."""
from typing import Any, Dict, List, Mapping, Optional

import requests
from pydantic import Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
INTRO_BLURB = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)
PROMPT_FOR_GENERATION_FORMAT = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

class MosaicML(LLM):
    endpoint_url: str 
    inject_instruction_format: bool 
    model_kwargs: Optional[dict] 
    retry_sleep: float 
    mosaicml_api_token: Optional[str]

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        pass

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        pass

    @property
    def _llm_type(self) -> str:
        pass

    def _transform_prompt(self, prompt: str) -> str:
        pass

    def _call(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        is_retry: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        responses = []
        for prompt in prompts:
            # Existing logic of _call method goes here
            # response = ...
            responses.append(response)
        return responses
]]>


class MosaicML(LLM):
    endpoint_url: str = (
        "https://models.hosted-on.mosaicml.hosting/mpt-7b-instruct/v1/predict"
    )
    inject_instruction_format: bool = False
    model_kwargs: Optional[dict] = None
    retry_sleep: float = 1.0
    mosaicml_api_token: Optional[str] = None

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        mosaicml_api_token = get_from_dict_or_env(
            values, "mosaicml_api_token", "MOSAICML_API_TOKEN"
        )
        values["mosaicml_api_token"] = mosaicml_api_token
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint_url": self.endpoint_url},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        return "mosaic"

    def _transform_prompt(self, prompt: str) -> str:
        if self.inject_instruction_format:
            prompt = PROMPT_FOR_GENERATION_FORMAT.format(
                instruction=prompt,
            )
        return prompt

    def _call(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        is_retry: bool = False,
        **kwargs: Any,
    ) -> List[str]:
        _model_kwargs = self.model_kwargs or {}

        responses = []
        for prompt in prompts:
            prompt = self._transform_prompt(prompt)

            payload = {"inputs": [prompt]}
            payload.update(_model_kwargs)
            payload.update(kwargs)

            headers = {
                "Authorization": f"{self.mosaicml_api_token}",
                "Content-Type": "application/json",
            }

            try:
                response = requests.post(self.endpoint_url, headers=headers, json=payload)
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Error raised by inference endpoint: {e}")

            try:
                parsed_response = response.json()

                if "error" in parsed_response:
                    if (
                        not is_retry
                        and "rate limit exceeded" in parsed_response["error"].lower()
                    ):
                        import time

                        time.sleep(self.retry_sleep)

                        return self._call(prompts, stop, run_manager, is_retry=True)

                    raise ValueError(
                        f"Error raised by inference API: {parsed_response['error']}"
                    )

                if isinstance(parsed_response, dict):
                    output_keys = ["data", "output", "outputs"]
                    for key in output_keys:
                        if key in parsed_response:
                            output_item = parsed_response[key]
                            break
                    else:
                        raise ValueError(
                            f"No valid key ({', '.join(output_keys)}) in response:"
                            f" {parsed_response}"
                        )
                    if isinstance(output_item, list):
                        text = output_item[0]
                    else:
                        text = output_item
                elif isinstance(parsed_response, list):
                    first_item = parsed_response[0]
                    if isinstance(first_item, str):
                        text = first_item
                    elif isinstance(first_item, dict):
                        if "output" in parsed_response:
                            text = first_item["output"]
                        else:
                            raise ValueError(
                                f"No key data or output in response: {parsed_response}"
                            )
                    else:
                        raise ValueError(f"Unexpected response format: {parsed_response}")
                else:
                    raise ValueError(f"Unexpected response type: {parsed_response}")

                text = text[len(prompt) :]

            except requests.exceptions.JSONDecodeError as e:
                raise ValueError(
                    f"Error raised by inference API: {e}.\nResponse: {response.text}"
                )

            if stop is not None:
                text = enforce_stop_tokens(text, stop)
            responses.append(text)

        return responses
