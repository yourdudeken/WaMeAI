from torch.optim import AdamW

model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(5):
    for input_ids, output_ids in tokenized_conversations:
        labels = output_ids.clone()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")