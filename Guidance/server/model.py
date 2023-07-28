
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


model_id = "mosaicml/mpt-7b-8k-instruct"
config = AutoConfig.from_pretrained(model_id,  trust_remote_code=True)

personal_tokenizer = AutoTokenizer.from_pretrained(model_id)

personal_model = AutoModelForCausalLM.from_pretrained(model_id,
                                                      cache_dir='/home/nero/Projects/shared/models',
                                                      config=config,
                                                      torch_dtype=torch.float16,
                                                      device_map="auto",
                                                      trust_remote_code=True,
                                                      load_in_8bit=True)


personal_model.eval()
   


