import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Configuration
BASE_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ADAPTER_PATH = "../Llama-3.1-Comedy-Adapter-Lables" 

def load_model():
    print("Loading base model & adapter...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    return tokenizer, model

def generate_text(tokenizer, model, input_text):
    system_prompt = (
        "You are a professional stand-up comedian. "
        "Generate a stand-up comedy transcript. "
        "Each component must have a type: Setup, Incongruity, Punchline, or Callback.\n"
        "Choose the most appropriate type for each component based on context.\n"
        "Follow a format of { (TYPE): \"SENTENCE\" }."
    )
    user_prompt = f"Generate a five-minute stand-up comedy transcript, labeling each part.\n\nTranscript:\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 1. Apply chat template to generate input tokens
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)

    # [Fix] Manually create an attention mask
    # Since the input is a single sequence without padding, we set all values to 1
    attention_mask = torch.ones_like(input_ids)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            # [Fix] Explicitly pass attention_mask and pad_token_id to suppress warnings
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,
        )

    # Decode the generated response (skipping input tokens)
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)
    
def parse_and_save_jsonl(raw_text, output_filename="inference_result.jsonl"):
    """
    Parses the text output and appends it as a JSONL line.
    """
    pattern = r'\((Setup|Incongruity|Punchline|Callback)\):\s*"?([^"\n]+)"?'
    parsed_items = []
    
    lines = raw_text.split('\n')
    for line in lines:
        match = re.search(pattern, line.strip(), re.IGNORECASE)
        if match:
            parsed_items.append({
                "type": match.group(1).capitalize(),
                "text": match.group(2).strip()
            })
            
    # Save as JSONL (Append mode 'a' is useful for accumulating results)
    with open(output_filename, 'a', encoding='utf-8') as f:
        # Wrap in a structure similar to training data or just the list
        output_entry = {"result": parsed_items} 
        f.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
    
    print(f"Result appended to {output_filename}")

if __name__ == "__main__":
    test_text = ""
    
    tokenizer, model = load_model()
    print("\n--- Generating... ---\n")
    
    raw_result = generate_text(tokenizer, model, test_text)
    print(raw_result)
    
    # Save output to JSONL
    parse_and_save_jsonl(raw_result)