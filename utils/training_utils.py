import torch

def train_client(model, loader, device, optimizer, loss_fn, num_epochs=1, log=True, mix_transform=None):
    model.train()
    history = []

    has_aux = hasattr(model, "get_aux_loss")

    for epoch in range(num_epochs):
        running_base = 0.0
        running_aux = 0.0
        running_total = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mix_transform is not None:
                x, y = mix_transform(x, y)

            optimizer.zero_grad(set_to_none=True)

            if has_aux:
                logits, aux = model(x, return_aux=True)
                base_loss = loss_fn(logits, y)
                total_loss = base_loss + aux
                aux_val = aux.detach()
            else:
                logits = model(x)
                base_loss = loss_fn(logits, y)
                total_loss = base_loss
                aux_val = None

            total_loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_base += base_loss.item() * bs
            running_total += total_loss.item() * bs
            if aux_val is not None:
                running_aux += aux_val.item() * bs

            if y.ndim == 2:          # soft labels from MixUp/CutMix
                y_hard = y.argmax(dim=1)
            else:
                y_hard = y
            correct += (logits.argmax(dim=1) == y_hard).sum().item()
            total += bs

        epoch_base = running_base / total
        epoch_total = running_total / total
        epoch_acc = correct / total
        epoch_aux = (running_aux / total) if has_aux else None

        m = {
            "epoch": epoch + 1,
            "total_loss": epoch_total,          
            "acc": epoch_acc,
            "samples": total,
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
        running_base = 0.0
        running_aux = 0.0
        running_prox = 0.0
        running_total = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mix_transform is not None:
                x, y = mix_transform(x, y)

            optimizer.zero_grad(set_to_none=True)

            if has_aux:
                logits, aux = model(x, return_aux=True)
                base_loss = loss_fn(logits, y)
                total_loss = base_loss + aux
                aux_val = aux.detach()
            else:
                logits = model(x)
                base_loss = loss_fn(logits, y)
                total_loss = base_loss
                aux_val = None
            
            #FedProx loss added
            prox = 0.0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    # accumulate in fp32 for numerical stability
                    diff = p.float() - w0[name].float()
                    prox = prox + diff.pow(2).sum()
            prox_term = 0.5 * mu * prox
            total_loss = total_loss + prox_term
            prox_val = prox_term.detach()

            total_loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_base += base_loss.item() * bs
            running_prox += prox_val.item() * bs
            running_total += total_loss.item() * bs
            if aux_val is not None:
                running_aux += aux_val.item() * bs

            if y.ndim == 2:  #soft labels from MixUp/CutMix
                y_hard = y.argmax(dim=1)
            else:
                y_hard = y
            correct += (logits.argmax(dim=1) == y_hard).sum().item()
            total += bs

        epoch_base = running_base / total
        epoch_prox = running_prox / total
        epoch_total = running_total / total
        epoch_acc = correct / total
        epoch_aux = (running_aux / total) if has_aux else None

        m = {
            "epoch": epoch + 1,
            "prox_loss": epoch_prox,
            "base_loss" : epoch_base,
            "total_loss": epoch_total,          
            "acc": epoch_acc,
            "samples": total,
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

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += bs

    return {
        "loss": total_loss / total_samples,
        "acc": total_correct / total_samples,
        "samples": total_samples,
    }