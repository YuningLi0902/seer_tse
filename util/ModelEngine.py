from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from util.FileHelper import ReadFileContent
from util.Model import OpenAILM

# Keep these aliases consistent with OpenAILM.model_dict in util/Model.py.
CLOSE_SOURCE_MODEL_LIST = ["gpt5.2", "claude4.6sonnet", "deepseekv4flash"]


class ModelEngine:
    def __init__(self, args):
        self.args = args
        self.model_id = args.model
        if self.model_id in CLOSE_SOURCE_MODEL_LIST:
            api_key = ReadFileContent(args.key_file)
            self.engine = OpenAILM(self.model_id, api_key, args.temperature)
        else:
            raise ValueError("Unsupported model alias: {}. Update the model lists before running SEERS1.".format(self.model_id))

    def predict(self, inputs, use_history=False, system_message=None, return_consumed_tokens=False):
        if self.model_id in CLOSE_SOURCE_MODEL_LIST:
            response, consumed_tokens = self.engine.predict(inputs, use_history=use_history,
                                                            system_message=system_message)
            if return_consumed_tokens:
                return response, consumed_tokens
            else:
                return response
        return self.engine.predict(inputs)

    def cleanHistory(self):
        if self.model_id in CLOSE_SOURCE_MODEL_LIST:
            self.engine.cleanMessage()
            self.engine.resetConsumedTokens()

    def setSystemMessage(self, system_message):
        if self.model_id in CLOSE_SOURCE_MODEL_LIST:
            self.engine.setSystemMessage(system_message)

    def getConsumedTokens(self):
        if self.model_id in CLOSE_SOURCE_MODEL_LIST:
            return self.engine.getConsumedTokens()
