import os
import torch
import numpy as np
import ctranslate2
from transformers import AutoTokenizer

class Embedder:
    _instance = None
    
    def __new__(cls, model_name=None):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._tokenizer = None
            cls._instance._model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
            cls._instance._device = os.getenv("EMBEDDING_DEVICE", "cpu")
            cls._instance._model_save_path = os.getenv("EMBEDDING_MODEL_PATH", "bge_model_ctranslate2")
        return cls._instance
    
    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            print(f"Loading embedding model: {self._model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            
            if self._device == "cuda":
                self._model = ctranslate2.Encoder(
                    self._model_save_path, device=self._device, compute_type="float16"
                )
            else:
                self._model = ctranslate2.Encoder(self._model_save_path, device=self._device)
                
            print(f"Embedding model loaded successfully")
    
    def generate_embedding(self, text):

        self._load_model()
        
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids)
        
        output = self._model.forward_batch([tokens])
        
        last_hidden_state = output.last_hidden_state
        last_hidden_state = np.array(last_hidden_state)
        last_hidden_state = torch.as_tensor(last_hidden_state, device=self._device)[0]
        
        last_hidden_state = torch.nn.functional.normalize(last_hidden_state, p=2, dim=1)
        
        if self._device == "cuda":
            embeddings = last_hidden_state.detach().cpu().tolist()[0]
        else:
            embeddings = last_hidden_state.detach().tolist()[0]
            
        return embeddings 