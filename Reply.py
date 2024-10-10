model.eval()

def generate_reply(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    return reply

user_input = "Tell me a joke."
reply = generate_reply(user_input)
print(f"AI: {reply}")