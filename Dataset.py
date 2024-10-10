conversations = [
    {"input": "Hello, how are you?", "output": "I'm good, thank you! How can I help you today?"},
    {"input": "What's the weather like?", "output": "It's sunny and warm outside."},
    {"input": "Tell me a joke.", "output": "Why don't scientists trust atoms? Because they make up everything!"},
]

def tokenize_conversation(conv):
    input_ids = tokenizer.encode(conv["input"], return_tensors="pt")
    output_ids = tokenizer.encode(conv["output"], return_tensors="pt")
    return input_ids, output_ids

tokenized_conversations = [tokenize_conversation(conv) for conv in conversations]