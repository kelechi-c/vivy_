"""
AR trainer script
"""

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from accelerate import Accelerator, notebook_launcher
from torch.cuda.amp import GradScaler
from safetensors.torch import save_model
import torchaudio
from transformers import LlamaModel
import wandb
import os
import gc
from time import time
import datetime
from tqdm.auto import tqdm
from .ar_model import tiny_llama, tokenizer
from .ar_utils import seed_everything, config, train_configs
from .dataloader import mini_train_loader

seed_everything()  # set environment seed

# training definitions
model = tiny_llama

loss_fn = nn.CrossEntropyLoss()  # loss function
optimizer = optim.AdamW(model.parameters(), lr=train_configs.lr)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,  # restart every 1000 steps
    T_mult=1,
)
scaler = GradScaler()

# configure accelerate
accelerator = Accelerator()
audio_model, class_dataloader, optimizer, scheduler = accelerator.prepare(
    # cofnigure modules for training
    model,
    mini_train_loader,
    optimizer,
    scheduler,
)


def count_params(model: nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count


print(f"model parameters (training) = {count_params(audio_model)}")


def logger(model) -> None:
    wandb.login()
    wandb.init(project="canary_v1", name="audiogen-1-tinyllama-classtext")
    wandb.watch(model)


logger(model)


@torch.no_grad
def epoch_sample(model: LlamaModel = model, prompt_class="classical"):
    tokenized = tokenizer.encode(prompt_class)
    sample_tokens = model.generate(**tokenized)
    now = datetime.datetime.now()
    filename = now.strftime("%m%d_%H%M%S") + ".wav"
    file_name = os.path.join(config.outpath, filename)
    print("saved: ", file_name)
    torchaudio.save(file_name, sample_tokens, data_configs.sample_rate)

    return filename


def trainer(
    model=model, train_loader=mini_train_loader, epoch_count=train_configs.epochs
):
    model.train()
    model.to(config.device)

    train_loss = 0.0
    # training loop
    for epoch in tqdm(range(epoch_count)):
        print(f"training for epoch {epoch+1}")
        start_time = time()
        optimizer.zero_grad()  # clear gradient graph

        for step, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()  # clear gradient graph

            input_tokens = batch["input_ids"].to(config.device)
            attn_mask = batch["attention_mask"].to(config.device)

            assert (
                input_tokens.max() < model.config.vocab_size
            ), f"Input contains token ID {input_tokens.max().item()} which is >= vocab size {model.config.vocab_size}"
            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=input_tokens.long().squeeze(),  # .squeeze(),
                    attention_mask=attn_mask.long().squeeze(),  # .squeeze(),
                    #                     labels=input_tokens.long().squeeze(),
                )[0]
                outputs = model.lm_head(outputs)

                # clear memory
                clearmem()

                # slice tensors, due to 'next-token prediction' objective
                # all except last token
                output_tensor = outputs[..., :-1, :].contiguous()
                # all except the first token
                targets = input_tokens[..., 1:].contiguous()
                shift_mask = attn_mask[..., 1:].contiguous()

                model_output = output_tensor.view(-1, output_tensor.size(-1))
                targets = targets.view(-1)

                # compute loss for step
                step_loss = loss_fn(model_output, targets)
                clearmem()

                total_tokens = shift_mask.sum()
                step_loss = step_loss.sum() / (total_tokens + 1e-8)

                # Scale loss by accumulation steps
                train_loss = step_loss / train_configs.grad_steps  # Normalize the loss

                print(f"step {step}: loss {step_loss:.4f}")
                wandb.log({"step_loss": step_loss})

                clearmem()
            # optimizer.step()

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(train_loss).backward()

            clearmem()

            if (step + 1) % train_configs.grad_steps == 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

            if step % 5 == 0:
                wandb.log({"train_loss": train_loss})

            if (step % 500) == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss": train_loss,
                }

                # save checkpoint
                torch.save(checkpoint, f"kaminari_mini_check_{epoch}.pth")

            if (step % 50) == 0:
                # log audio sample to WandB
                try:
                    test_sample_file = epoch_sample(model)
                    wandb.log(
                        {
                            "audio_sample": wandb.Audio(
                                test_sample_file,
                                caption=f"test_audio_track_{step}",
                                sample_rate=data_configs.sample_rate,
                            )
                        }
                    )
                except Exception as e:
                    print(f"error logging sample: {e}")

        #         scheduler.step()

        gc.collect()
        epoch_time = time() - start_time

        print(f"Epoch {epoch} of {epoch_count}, train_loss: {train_loss:.4f}")

        print(f"Epoch @ {epoch} complete in {epoch_time}!")

    print(
        f"End metrics for run of {epoch_count}, train_loss: {train_loss:.4f}")

    save_model(model, train_configs.sft_file)  # save to .safetensors file
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        #                     "scheduler_state": scheduler.state_dict(),
        "loss": train_loss,
    }
    torch.save(checkpoint, f"check_{train_configs.model_file}")

    torch.save(model.state_dict(), f"{train_configs.model_file}")

    return model


model = trainer()

# notebook_launcher(trainer_wrapper, num_processes=2)
# sayonara
