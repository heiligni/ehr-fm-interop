from huggingface_hub import HfFolder
import femr
from femr.ontology import Ontology
from typing import Optional

from models.processor import FEMRBatchProcessor
from models.transformer import FEMRModel

MODEL_NAME = "StanfordShahLab/clmbr-t-base"

def init_hf_hub():
    # To use the code you have to set the API token
    api_token = None
    HfFolder.save_token(api_token)  

def load_clmbr_t_base(ontology: Optional[Ontology]):
    init_hf_hub()
    
    # Load tokenizer / batch loader
    tokenizer = get_tokenizer(ontology)
    batch_processor = FEMRBatchProcessor(tokenizer)

    # Load model
    model = FEMRModel.from_pretrained(MODEL_NAME)
    return (model, tokenizer, batch_processor)

def get_tokenizer(ontology: Optional[Ontology]):
    init_hf_hub()
    
    return femr.models.tokenizer.FEMRTokenizer.from_pretrained(MODEL_NAME, ontology)
