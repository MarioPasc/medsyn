import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


@torch.no_grad()
def visualize_noising_process(
    x0: torch.Tensor,
    scheduler: DDPMScheduler,
    num_steps: int = 10,
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Visualize the forward noising process at evenly spaced timesteps.
    
    Args:
        x0: Clean image [1, C, H, W]
        scheduler: DDPM scheduler
        num_steps: Number of intermediate steps to visualize
        device: Device
    
    Returns:
        Tensor of shape [num_steps+1, C, H, W] showing progressive noising
    """
    x0 = x0.to(device)
    total_timesteps = scheduler.config.num_train_timesteps  # type: ignore[attr-defined]
    timesteps = torch.linspace(0, total_timesteps - 1, num_steps, dtype=torch.long, device=device)
    
    noised_images = [x0.cpu()]
    noise = torch.randn_like(x0)
    
    for t in timesteps:
        t_batch = t.unsqueeze(0)
        x_t = scheduler.add_noise(x0, noise, t_batch)
        noised_images.append(x_t.cpu())
    
    return torch.cat(noised_images, dim=0)

@torch.no_grad()
def visualize_denoising_process(
    model: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple,
    class_label: torch.Tensor,
    num_steps: int = 10,
    device: torch.device = torch.device("cuda"),
    guidance_scale: float = 1.0
) -> torch.Tensor:
    """
    Visualize the reverse denoising process from pure noise to clean image.

    Args:
        model: DDPM model
        scheduler: DDPM scheduler
        shape: Image shape (C, H, W)
        class_label: Class label for conditional generation
        num_steps: Number of intermediate steps to visualize (>= 1)
        device: Device
        guidance_scale: Classifier-free guidance scale

    Returns:
        Tensor showing progressive denoising: [initial_noise, *intermediate_steps, final_image]
        Shape: [num_steps+2, C, H, W] with initial noise + num_steps + final frame
    """
    model.eval()

    # Start from pure noise
    x_t = torch.randn((1, *shape), device=device)
    class_label = class_label.to(device)

    total_timesteps = scheduler.config.num_train_timesteps  # type: ignore[attr-defined]
    scheduler.set_timesteps(total_timesteps)
    # Move scheduler timesteps to device to avoid device mismatch in scheduler.step()
    scheduler.timesteps = scheduler.timesteps.to(device)

    # Ensure at least 1 intermediate step to maintain schedule compatibility
    save_indices = set(torch.linspace(
        0, len(scheduler.timesteps) - 1,
        max(1, num_steps),  # ensure >= 1 for proper step scheduling
        dtype=torch.long
    ).tolist())

    frames = [x_t.cpu()]  # keep initial noise for context

    for i, t in enumerate(scheduler.timesteps):
        # Predict noise
        t_batch = t.unsqueeze(0)

        if guidance_scale == 1.0:
            noise_pred = model(x_t, t_batch, class_label)
        else:
            # Classifier-free guidance
            noise_pred_cond = model(x_t, t_batch, class_label)
            noise_pred_uncond = model(x_t, t_batch, None)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Denoise one step
        x_t = scheduler.step(noise_pred, t, x_t).prev_sample  # type: ignore[union-attr]

        if i in save_indices:
            frames.append(x_t.detach().cpu())

    # Safeguard: ensure we always have the final denoised image
    if len(frames) == 1 or frames[-1] is not x_t.cpu():
        frames.append(x_t.detach().cpu())

    return torch.cat(frames, dim=0)

@torch.no_grad()
def visualize_multistep_reconstruction(
    model: nn.Module,
    x0: torch.Tensor,
    scheduler: DDPMScheduler,
    class_label: torch.Tensor,
    timesteps: list[int],
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    Show x0 reconstruction quality at different timesteps.
    
    Args:
        model: DDPM model
        x0: Clean image [1, C, H, W]
        scheduler: DDPM scheduler
        class_label: Class label
        timesteps: List of timesteps to visualize
        device: Device
    
    Returns:
        Tensor [len(timesteps)+1, C, H, W] with original + reconstructions
    """
    model.eval()
    x0 = x0.to(device)
    class_label = class_label.to(device)
    
    reconstructions = [x0.cpu()]
    noise = torch.randn_like(x0)
    
    for t_val in timesteps:
        t = torch.tensor([t_val], dtype=torch.long, device=device)
        
        # Add noise
        x_t = scheduler.add_noise(x0, noise, t)  # type: ignore[arg-type]
        
        # Predict noise
        noise_pred = model(x_t, t, class_label)
        
        # Reconstruct x0
        sqrt_alpha_prod = scheduler.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alpha_prod = (1 - scheduler.alphas_cumprod[t]).sqrt()
        x0_pred = (x_t - sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1) * noise_pred) / sqrt_alpha_prod.view(-1, 1, 1, 1)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        reconstructions.append(x0_pred.cpu())
    
    return torch.cat(reconstructions, dim=0)