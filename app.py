from transformers import (AutoModel, AutoTokenizer,  AutoModelForCausalLM,
                          Trainer, TrainingArguments, TrainerCallback,)
import torch

import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H6("Change the value in the text box to see callbacks in action!"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    html.Br(),
    html.Div(id='my-output'),

])

MODEL = "sberbank-ai/rugpt3medium_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
special_tokens = {
    "bos_token": "<|startoftext|>",
    "pad_token": "<|pad|>",
    "sep_token": "<|sep|>",
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
checkpoint = torch.load("simplification.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)
MAX_LENGTH = 200
model.eval()


def add_tokens(input):
    return f"<|startoftext|>{input}<|sep|>"


def transform_sentence(sentence):
    sentence = add_tokens(sentence)
    input = tokenizer.encode(sentence, return_tensors="pt")
    with torch.no_grad():
        sample_outputs = model.generate(
            input,
            do_sample=True,
            top_k=50,
            max_length=MAX_LENGTH,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=5
        ).detach().cpu()
    results = []
    for sample in sample_outputs:
        res = (tokenizer.decode(sample, skip_special_tokens=False)
               .split("<|sep|>")[1]
               .replace("<|pad|>", "")
               .replace("<|endoftext|>", ""))
        results.append(res)
    return results


if __name__ == '__main__':
    app.run_server(debug=True)
