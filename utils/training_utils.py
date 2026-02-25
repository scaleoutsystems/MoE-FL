import torch
from torch.amp import autocast, GradScaler

def train_client(model, loader, device, optimizer, loss_fn, num_epochs=1, log=True, mix_transform=None):
    scaler = GradScaler(device=device)
    
    model.train()
    history = []

    has_aux = hasattr(model, "get_aux_loss")

    for epoch in range(num_epochs):
        running_base = torch.zeros((), device=device)
        running_aux = torch.zeros((), device=device)
        running_total = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        total = torch.zeros((), device=device, dtype=torch.long)

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mix_transform is not None:
                x, y = mix_transform(x, y)

            optimizer.zero_grad(set_to_none=True)
            
            #Tesla T4 uses float16
            with autocast(device_type="cuda", dtype=torch.float16):
                if has_aux:
                    logits, aux = model(x, return_aux=True)
                    base_loss = loss_fn(logits, y)
                    total_loss = base_loss + aux
                else:
                    logits = model(x)
                    base_loss = loss_fn(logits, y)
                    total_loss = base_loss
                    
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            running_base  += base_loss.detach() * bs
            running_total += total_loss.detach() * bs
            if has_aux:
                running_aux += aux.detach() * bs

            if y.ndim == 2:          # soft labels from MixUp/CutMix
                y_hard = y.argmax(dim=1)
            else:
                y_hard = y
            correct += (logits.argmax(dim=1) == y_hard).sum()
            total += bs

        epoch_base = (running_base / total).item()
        epoch_total = (running_total / total).item()
        epoch_acc = (correct / total).item()
        epoch_aux = (running_aux / total).item() if has_aux else None

        m = {
            "epoch": epoch + 1,
            "total_loss": epoch_total,          
            "acc": epoch_acc,
        }
        
        if has_aux:
            m["base_loss"] = epoch_base
            m["aux_loss"] = epoch_aux

        history.append(m)

        if log:
            if epoch_aux is None:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] - "
                    f"loss: {epoch_total:.4f} - acc: {epoch_acc:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] - "
                    f"base loss: {epoch_base:.4f} - aux loss: {epoch_aux:.6f} - "
                    f"total loss: {epoch_total:.4f} - acc: {epoch_acc:.4f}"
                )

    return history


def train_client_fedprox(model, loader, device, optimizer, loss_fn, mu, num_epochs=1, log=True, mix_transform=None):
    scaler = GradScaler(device=device)
    model.train()
    history = []

    has_aux = hasattr(model, "get_aux_loss")
    
    #Save global model for fedprox
    #this is the global model as no updates has occured yet
    w0 = {name: p.detach().clone()   # clone so it doesn't change as p updates
        for name, p in model.named_parameters()
        if p.requires_grad
        }

    for epoch in range(num_epochs):
        running_base = torch.zeros((), device=device)
        running_aux = torch.zeros((), device=device)
        running_prox = torch.zeros((), device=device)
        running_total = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)
        total = torch.zeros((), device=device, dtype=torch.long)

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mix_transform is not None:
                x, y = mix_transform(x, y)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.float16):
                if has_aux:
                    logits, aux = model(x, return_aux=True)
                    base_loss = loss_fn(logits, y)
                    total_loss = base_loss + aux
                else:
                    logits = model(x)
                    base_loss = loss_fn(logits, y)
                    total_loss = base_loss
            
            #FedProx loss added
            prox = torch.zeros((), device=device, dtype=torch.float32)
            for name, p in model.named_parameters():
                if p.requires_grad:
                    # accumulate in fp32 for numerical stability
                    diff = p.float() - w0[name].float()
                    prox = prox + diff.pow(2).sum()
            prox_term = 0.5 * mu * prox
            total_loss = total_loss + prox_term
            prox_val = prox_term.detach()

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = x.size(0)
            running_base += base_loss.detach() * bs
            running_prox += prox_val.detach() * bs
            running_total += total_loss.detach() * bs
            if has_aux:
                running_aux += aux.detach() * bs

            if y.ndim == 2:  #soft labels from MixUp/CutMix
                y_hard = y.argmax(dim=1)
            else:
                y_hard = y
            correct += (logits.argmax(dim=1) == y_hard).sum()
            total += bs

        epoch_base = (running_base / total).item()
        epoch_prox = (running_prox / total).item()
        epoch_total = (running_total / total).item()
        epoch_acc = (correct / total).item()
        epoch_aux = (running_aux / total).item() if has_aux else None

        m = {
            "epoch": epoch + 1,
            "prox_loss": epoch_prox,
            "base_loss" : epoch_base,
            "total_loss": epoch_total,          
            "acc": epoch_acc,
        }
        
        if has_aux:
            m["aux_loss"] = epoch_aux

        history.append(m)

        if log:
            if epoch_aux is None:
                 print(
                    f"Epoch [{epoch+1}/{num_epochs}] - base loss: {epoch_base:.4f} - "
                    f"Prox loss: {epoch_prox:.4f} - "
                    f"total loss: {epoch_total:.4f} - acc: {epoch_acc:.4f}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] - base loss: {epoch_base:.4f} - "
                    f"Prox loss: {epoch_prox:.4f} - aux loss: {epoch_aux:.6f} - "
                    f"total loss: {epoch_total:.4f} - acc: {epoch_acc:.4f}"
                )

    return history

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = torch.zeros((), device=device, dtype=torch.long)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.detach() * bs
        total_correct += (logits.argmax(dim=1) == y).sum()
        total_samples += bs

    return {
        "loss": (total_loss / total_samples).item(),
        "acc": (total_correct.float() / total_samples).item(),
    }