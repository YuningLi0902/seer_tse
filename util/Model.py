from openai import OpenAI


class CSLM(object):
    def __init__(self, model_name, api_key, temperature=1.0, max_tokens=8192):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.message = []
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0

    def cleanMessage(self):
        self.message.clear()

    def setSystemMessage(self, system_message):
        self.message.append({"role": "system", "content": system_message})

    def setUserMessage(self, user_message):
        self.message.append({"role": "user", "content": user_message})

    def getConsumedTokens(self):
        return [self.session_prompt_tokens, self.session_completion_tokens, self.session_total_tokens]

    def resetConsumedTokens(self):
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0


class OpenAILM(CSLM):
    def __init__(self, model_name, api_key, temperature=1.0, max_tokens=8192):
        print("Initializing an OpenAI chat model: {} ...".format(model_name))
        super().__init__(model_name, api_key, temperature, max_tokens)
        # Fill in endpoint URLs and provider model IDs before making API calls.
        self.model_dict = {
            "gpt5.2": ["YOUR_OPENAI_COMPATIBLE_BASE_URL", "YOUR_GPT_MODEL_ID"],
            "claude4.6sonnet": ["YOUR_OPENAI_COMPATIBLE_BASE_URL", "YOUR_CLAUDE_MODEL_ID"],
            "deepseekv4flash": ["YOUR_OPENAI_COMPATIBLE_BASE_URL", "YOUR_DEEPSEEK_MODEL_ID"]
        }
        if self.model_name not in self.model_dict:
            raise ValueError("Unknown model alias: {}. Update util/Model.py and util/ModelEngine.py.".format(self.model_name))
        base_url, provider_model = self.model_dict[self.model_name]
        if not base_url or not provider_model or base_url.startswith("YOUR_") or provider_model.startswith("YOUR_"):
            raise ValueError("Fill in the endpoint and model ID in util/Model.py before running SEERS1.")
        self.model = OpenAI(api_key=api_key, base_url=self.model_dict[self.model_name][0])

    def predict(self, inputs, use_history=False, stop_words=None, system_message=None, max_retry=3):
        if stop_words is None:
            stop_words = []
        new_message = []
        if system_message:
            new_message.append({"role": "system", "content": system_message})
        if type(inputs) is list:
            for input_str in inputs:
                new_message.append({"role": "user", "content": input_str})
        else:
            new_message.append({"role": "user", "content": inputs})
        if use_history:
            self.message.extend(new_message)
            message = self.message
        else:
            message = new_message
        try_num = 0
        response = None
        response_message = ""
        while try_num < max_retry and response_message == "":
            try_num += 1
            response = self.model.chat.completions.create(
                model=self.model_dict[self.model_name][1],
                messages=message,
                stream=False,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop_words
            )
            response_message = response.choices[0].message.content
        assert response_message != ""
        consumed_token_last_round = [response.usage.prompt_tokens, response.usage.completion_tokens,
                                     response.usage.total_tokens]
        if use_history:
            self.message.append({"role": "assistant", "content": response_message})
            self.session_prompt_tokens += consumed_token_last_round[0]
            self.session_completion_tokens += consumed_token_last_round[1]
            self.session_total_tokens += consumed_token_last_round[2]
        return response_message, consumed_token_last_round
