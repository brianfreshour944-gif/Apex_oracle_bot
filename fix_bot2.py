# Fix bot.py indentation issue
old_section = """                              x = torch.tensor(data_scaled).unsqueeze(0).to(device)
                              label_tensor = torch.tensor([[1.0 if realized_pnl > 0 else 0.0]], dtype=torch.float32).to(device)
                              
                              model.train()
                              model.zero_grad()
                              logits = model(x)
                              loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_tensor)
                              loss.backward()
                              
                              # Learning rate schedule with warmup + cosine annealing
                              global _transformer_online_updates, _transformer_online_lr_schedule, _transformer_online_lr_base, _transformer_online_lr_min, _transformer_online_warmup_steps, _transformer_online_total_steps
                              
                              step = _transformer_online_updates
                              
                              if step < _transformer_online_warmup_steps:
                                  # Linear warmup
                                  lr = _transformer_online_lr_base * (step + 1) / _transformer_online_warmup_steps
                              else:
                                  progress = min((step - _transformer_online_warmup_steps) / (_transformer_online_total_steps - _transformer_online_warmup_steps), 1.0)
                                  if _transformer_online_lr_schedule == "cosine":
                                      # Cosine annealing to min LR
                                      lr = _transformer_online_lr_min + 0.5 * (_transformer_online_lr_base - _transformer_online_lr_min) * (1 + math.cos(math.pi * progress))
                                  elif _transformer_online_lr_schedule == "linear":
                                      # Linear decay to min LR
                                      lr = _transformer_online_lr_base - (_transformer_online_lr_base - _transformer_online_lr_min) * progress
                                  else:  # constant
                                      lr = _transformer_online_lr_base
                              
                              _transformer_online_updates += 1
                              with torch.no_grad():
                                  for p in model.parameters():
                                      if p.grad is not None:
                                          p.data -= lr * p.grad
                              model.eval()"""

new_section = """                              x = torch.tensor(data_scaled).unsqueeze(0).to(device)
                              label_tensor = torch.tensor([[1.0 if realized_pnl > 0 else 0.0]], dtype=torch.float32).to(device)
                              
                              # Protect model train/eval state from concurrent access
                              with _model_inference_lock:
                                  model.train()
                                  model.zero_grad()
                                  logits = model(x)
                                  loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_tensor)
                                  loss.backward()
                                  
                                  # Learning rate schedule with warmup + cosine annealing
                                  global _transformer_online_updates, _transformer_online_lr_schedule, _transformer_online_lr_base, _transformer_online_lr_min, _transformer_online_warmup_steps, _transformer_online_total_steps
                                  
                                  step = _transformer_online_updates
                                  
                                  if step < _transformer_online_warmup_steps:
                                      # Linear warmup
                                      lr = _transformer_online_lr_base * (step + 1) / _transformer_online_warmup_steps
                                  else:
                                      progress = min((step - _transformer_online_warmup_steps) / (_transformer_online_total_steps - _transformer_online_warmup_steps), 1.0)
                                      if _transformer_online_lr_schedule == "cosine":
                                          # Cosine annealing to min LR
                                          lr = _transformer_online_lr_min + 0.5 * (_transformer_online_lr_base - _transformer_online_lr_min) * (1 + math.cos(math.pi * progress))
                                      elif _transformer_online_lr_schedule == "linear":
                                          # Linear decay to min LR
                                          lr = _transformer_online_lr_base - (_transformer_online_lr_base - _transformer_online_lr_min) * progress
                                      else:  # constant
                                          lr = _transformer_online_lr_base
                                      
                                      _transformer_online_updates += 1
                                      with torch.no_grad():
                                          for p in model.parameters():
                                              if p.grad is not None:
                                                  p.data -= lr * p.grad
                                      model.eval()"""

with open('src/bot.py', 'r') as f:
    content = f.read()

if old_section in content:
    content = content.replace(old_section, new_section)
    with open('src/bot.py', 'w') as f:
        f.write(content)
    print("Replacement successful!")
else:
    print("Old section not found!")
    # Try to find what's there
    with open('src/bot.py', 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'x = torch.tensor(data_scaled).unsqueeze(0).to(device)' in line:
            print(f"Line {i}: indent={len(line) - len(line.lstrip())} {repr(line[:80])}")