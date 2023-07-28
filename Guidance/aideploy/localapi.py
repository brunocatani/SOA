from typing import Any, Dict, Generator, List, Optional
import requests
from langchain.llms.base import LLM
from IPython.display import HTML, display

#WRAPPER CUSTOMIZADO DA API DO TEXT.GENERATION.UI

#!!!!!!!!!NÃO MODIFICAR!!!!!!!!!!!


class TGWebUI(LLM):

    max_new_tokens: int = 500
    temperature: float = 0.3
    top_p: float = 1
    top_k: int = 40
    typical_p: float = 1.0
    repetition_penalty: float = 1.2
    encoder_repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    do_sample: bool = True
    seed: int = -1
    penalty_alpha: float = 0.0
    num_beams: int = 1
    length_penalty: float = 1.0   
    early_stopping: bool = False
    truncation_length: int = 2048
    stop: List[str] = None
    add_bos_token: bool = True  
    ban_eos_token: bool = False
    skip_special_tokens: bool = True
    
    api_host: str = "localhost"
    api_port: int = 5000
    api_streaming_port: int = 5005
    use_https: bool = False
        
    verbose: bool = False

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "encoder_repetition_penalty": self.encoder_repetition_penalty,
            "typical_p": self.typical_p,
            "min_length": self.min_length,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "penalty_alpha": self.penalty_alpha,
            "early_stopping": self.early_stopping,
            "seed": self.seed,
            "add_bos_token": self.add_bos_token,
            "ban_eos_token": self.ban_eos_token,
            "truncation_length": self.truncation_length,
            "skip_special_tokens": self.skip_special_tokens,
            "do_sample": self.do_sample,
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return self._default_params

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "text-generation-webui"

    def _get_parameters(self, stop: Optional[List[str]] = None) -> Dict[str, Any]:

        params = self._default_params
        params["stop"] = self.stop or stop or []
        params["stopping_strings"] = params["stop"]

        return params
    
    def get_base_url(self):
        proto = "https" if self.use_https else "http"
        return f"{proto}://{self.api_host}:{self.api_port}"
    
    def get_model_name(self):
        URI = f'{self.get_base_url()}/api/v1/model'
        response = requests.get(URI)
        return response.json()['result']

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Call the text-generation-webui API and return the output.

        Args:
            prompt: The prompt to use for generation.
            stop: A list of strings to stop generation when encountered.

        Returns:
            The generated text.

        """
        request = self._get_parameters(stop) 
        generations = []

        if self.verbose:
            print(f"<<<{prompt}>>>")
        
        
        request['prompt'] = prompt

        URI = f'{self.get_base_url()}/api/v1/generate'
        response = requests.post(URI, json=request)

        if response.status_code == 200:
            result = response.json()['results'][0]['text']

            if self.verbose:
                print(f"<<<{result}>>>")

            return result


