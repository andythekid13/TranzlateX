from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the NLLB-200 Model
MODEL_NAME = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def translate_text(text, src_lang, tgt_lang):
    """Translates text from source to target language using NLLB-200."""
    model_inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(f'[{tgt_lang}]'))
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
