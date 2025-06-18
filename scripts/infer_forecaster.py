import argparse
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from fingpt.FinGPT_Forecaster.data_infererence_fetch import (
    fetch_all_data,
    get_all_prompts_online,
    get_curday,
)

SYSTEM_PROMPT = (
    "You are a seasoned stock market analyst. Your task is to list the positive developments "
    "and potential concerns for companies based on relevant news and basic financials from the past weeks, "
    "then provide an analysis and prediction for the companies' stock price movement for the upcoming week. "
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ..."
    "\n\n[Prediction & Analysis]\nPrediction: ...\nAnalysis: ..."
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def load_model(hf_token=None):
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        token=hf_token,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token)
    model = PeftModel.from_pretrained(base_model, "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora")
    return model.eval(), tokenizer


def generate_prompt(symbol: str, date: str, weeks: int, with_basics: bool) -> str:
    data = fetch_all_data(symbol, date, n_weeks=weeks)
    _, prompt = get_all_prompts_online(symbol, data, date, with_basics=with_basics)
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Run FinGPT-Forecaster inference")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--date", default=get_curday(), help="Prediction start date (YYYY-MM-DD)")
    parser.add_argument("--weeks", type=int, default=3, help="Number of past weeks to use")
    parser.add_argument("--no_basics", action="store_true", help="Exclude basic financials")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"), help="HuggingFace access token")
    args = parser.parse_args()

    model, tokenizer = load_model(args.hf_token)

    prompt = generate_prompt(args.symbol, args.date, args.weeks, not args.no_basics)

    text = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    res = model.generate(
        **inputs,
        max_length=4096,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    answer = re.sub(r".*\[/INST\]\s*", "", output, flags=re.DOTALL)
    print(answer)


if __name__ == "__main__":
    main()
