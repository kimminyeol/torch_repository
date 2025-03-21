from transformers import get_linear_schedule_with_warmup

def get_scheduler(optimizer, num_training_steps, warmup_steps=0):
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )