import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, dataloader, epochs=2):
    criterion = nn.CTCLoss(blank=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (x, y) in enumerate(dataloader):
            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
            outputs = model(x)
            loss = criterion(outputs.log_softmax(2), ...)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
