from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()

def generate_reply(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    return reply

@app.route('/webhook', methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').strip()
    if incoming_msg:
        reply = generate_reply(incoming_msg)
    else:
        reply = "Sorry, I didn't understand that."

    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)

if __name__ == '__main__':
    app.run(debug=True)