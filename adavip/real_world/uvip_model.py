"""Joint-space L1 Teacher and S0 Student with the upstream UViP mechanisms."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from torch import Tensor, nn
from torchvision.models import resnet18


def load_unet(path: str) -> type:
    """Load the unchanged, pinned repository U-Net without simulator imports."""
    spec = importlib.util.spec_from_file_location("uvip_pinned_unet", Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConditionalUnet1D


class JointNormalizer(nn.Module):
    """Persist train-only reversible joint scales alongside every policy."""

    def __init__(self, width: int):
        super().__init__()
        for key in ("state", "action"):
            for suffix in ("scale", "offset", "min", "max"):
                self.register_buffer(key + "_" + suffix, torch.ones(width) if suffix == "scale" else torch.zeros(width))

    def fit(self, values: dict) -> None:
        """Install already fitted training-row statistics."""
        with torch.no_grad():
            if set(values) != set(self.state_dict()):
                raise ValueError("Normalizer schema changed")
            for key, value in values.items():
                tensor = torch.as_tensor(value, dtype=torch.float32)
                if tensor.shape != getattr(self, key).shape or not torch.isfinite(tensor).all():
                    raise ValueError("Invalid joint statistics")
                if key.endswith("_scale") and (tensor <= 0).any():
                    raise ValueError("Nonpositive scale")
                getattr(self, key).copy_(tensor)

    def normalize(self, key: str, value: Tensor) -> Tensor:
        return value * getattr(self, key + "_scale") + getattr(self, key + "_offset")

    def unnormalize(self, key: str, value: Tensor) -> Tensor:
        return (value - getattr(self, key + "_offset")) / getattr(self, key + "_scale")


class JointL1Encoder(nn.Module):
    """Independent ResNet18/GN histories, joint state and late frozen CLIP text."""

    def __init__(self, config: dict, backbone: nn.Module | None = None, image_width: int = 512):
        super().__init__()
        self.keys = list(config["image_keys"])
        self.history = len(config["history_offsets"])
        self.width = config["action_dim"]
        self.image_width = image_width
        # A backbone override is used only by fast synthetic regression tests.
        import copy
        if backbone is None:
            backbone = resnet18(weights=None)
            backbone.fc = nn.Identity()
            def replace(module: nn.Module) -> None:
                for name, child in module.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        setattr(module, name, nn.GroupNorm(math.gcd(32, child.num_features), child.num_features))
                    else:
                        replace(child)
            replace(backbone)
        self.images = nn.ModuleDict({key: copy.deepcopy(backbone) for key in self.keys})
        self.temporal = nn.ModuleDict({key: nn.Sequential(nn.Linear(self.history * image_width, 256),
                                                         nn.LayerNorm(256), nn.SiLU()) for key in self.keys})
        self.state = nn.Sequential(nn.Linear(self.history * self.width, 256), nn.LayerNorm(256), nn.SiLU())
        self.identity = nn.Parameter(torch.randn(len(self.keys) + 1, 256) * 0.02)
        count = len(self.keys) + 1
        self.fusion = nn.Sequential(nn.Linear(count * 256 + count, 256), nn.SiLU(), nn.Linear(256, 256))
        if "teacher_head_seed" in config:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config["teacher_head_seed"])
                self.language_head = nn.Sequential(nn.Linear(256 + 512, 256), nn.SiLU(), nn.Linear(256, 256))
        else:
            self.language_head = nn.Sequential(nn.Linear(256 + 512, 256), nn.SiLU(), nn.Linear(256, 256))

    def tokens(self, observations: dict, normalizer: JointNormalizer) -> Tensor:
        """Encode observed history only; masks are applied after identity addition."""
        batch = observations["state"].shape[0]
        tokens = []
        for key in self.keys:
            images = observations[key]
            if images.dtype != torch.uint8 or images.shape[:3] != (batch, self.history, 3):
                raise ValueError("Expected uint8 RGB [B,To,3,H,W]")
            features = self.images[key](images.flatten(0, 1).float().div(255).mul(2).sub(1))
            tokens.append(self.temporal[key](features.reshape(batch, -1)))
        state = observations["state"]
        if state.shape != (batch, self.history, self.width):
            raise ValueError("State joint width/history differs")
        tokens.append(self.state(normalizer.normalize("state", state).flatten(1)))
        return torch.stack(tokens, 1) + self.identity[None]

    def context(self, tokens: Tensor, language: Tensor, presence: Tensor) -> Tensor:
        """Remove entire sensor+identity tokens and flags, with no encoder bypass."""
        if presence.dtype != torch.bool or presence.shape != (len(tokens), len(self.keys) + 2):
            raise ValueError("Invalid input-group presence mask")
        visible = tokens.masked_fill(~presence[:, :-1, None], 0)
        state = self.fusion(torch.cat((visible.flatten(1), presence[:, :-1].to(tokens.dtype)), -1))
        language = language.masked_fill(~presence[:, -1:], 0)
        return self.language_head(torch.cat((state, language), -1))


class FilmHead(nn.Module):
    """Upstream identity-initialized, per-sensor 128 -> 128 -> 512 FiLM."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 512))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, token: Tensor, condition: Tensor) -> Tensor:
        gamma, beta = self.network(condition).chunk(2, -1)
        return (1 + gamma) * token + beta


class RealUvipPolicy(nn.Module):
    """Full-input L1 DP Teacher or S0 utility-conditioned Student for 7/14D joints."""

    def __init__(self, config: dict, student: bool = False, encoder: JointL1Encoder | None = None):
        super().__init__()
        self.config, self.student = config, student
        self.phase = "C" if student else "teacher"
        self.normalizer = JointNormalizer(config["action_dim"])
        self.encoder = encoder or JointL1Encoder(config)
        self.model = load_unet(config["upstream_unet"])(input_dim=config["action_dim"], global_cond_dim=256,
                                                       diffusion_step_embed_dim=128, down_dims=tuple(config["down_dims"]))
        self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                       variance_type="fixed_small", clip_sample=True, prediction_type="epsilon")
        groups = len(config["image_keys"]) + 2
        conditions = torch.ones(groups + 3, groups, dtype=torch.bool)
        for index in range(groups):
            conditions[index + 1, index] = False
        vision = len(config["vision_keys"])
        conditions[-2, :vision] = False
        conditions[-1, vision:vision + 2] = False
        self.register_buffer("conditions", conditions)
        utilities = groups + 2
        self.register_buffer("utility_rms", torch.ones(utilities))
        if student:
            self.utility_predictor = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, utilities))
            self.context_projection = nn.Sequential(nn.Linear(256, 64), nn.SiLU())
            self.utility_projection = nn.Sequential(nn.Linear(utilities, 64), nn.SiLU())
            self.film = nn.ModuleList([FilmHead() for _ in config["image_keys"]])
            self.language_projection = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU())
            self.query_projection = nn.Sequential(nn.Linear(utilities, 256), nn.SiLU(), nn.Linear(256, 256))
            nn.init.zeros_(self.query_projection[-1].weight)
            nn.init.zeros_(self.query_projection[-1].bias)
            self.attention = nn.MultiheadAttention(256, 4, batch_first=True)
            self.norm = nn.LayerNorm(256)
            self.ffn = nn.Sequential(nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 256))

    def initialize_from_teacher(self, teacher: RealUvipPolicy, rms: Tensor) -> None:
        """Inherit only encoder/normalizer, retaining the freshly created action U-Net."""
        if not self.student or teacher.student:
            raise ValueError("Expected Student and L1 Teacher")
        self.encoder.load_state_dict(teacher.encoder.state_dict(), strict=True)
        self.normalizer.load_state_dict(teacher.normalizer.state_dict(), strict=True)
        if rms.shape != self.utility_rms.shape or not torch.isfinite(rms).all() or (rms <= 0).any():
            raise ValueError("Invalid training-only utility RMS")
        self.utility_rms.copy_(rms)

    def set_normalizer(self, normalizer: JointNormalizer) -> None:
        """Copy persistent joint statistics using the repository policy interface."""
        self.normalizer.load_state_dict(normalizer.state_dict(), strict=True)

    def compute_loss(self, batch: dict) -> Tensor:
        """Return training loss for the currently selected phase."""
        return self.losses(batch, self.phase)["loss"]

    def set_phase(self, phase: str) -> None:
        """Freeze all but Utility in A; inherit/freeze encoder in B; unfreeze in C."""
        if phase not in ("A", "B", "C", "teacher"):
            raise ValueError(phase)
        self.phase = phase
        self.requires_grad_(phase != "A")
        self.normalizer.requires_grad_(False)
        if self.student:
            self.encoder.requires_grad_(phase == "C")
            self.utility_predictor.requires_grad_(True)
        elif phase != "teacher":
            raise ValueError("Teacher is trained action-only")

    def encode(self, observations: dict) -> dict[str, Tensor]:
        """Predict utility from detached context; detach utility on action paths."""
        tokens = self.encoder.tokens(observations, self.normalizer)
        presence = self.conditions[:1].expand(len(tokens), -1)
        context = self.encoder.context(tokens, observations["language"], presence)
        if not self.student:
            return dict(conditioning=context, context=context)
        utility = self.utility_predictor(context.detach())
        detached = utility.detach()
        film_condition = torch.cat((self.context_projection(context), self.utility_projection(detached)), -1)
        modulated = [head(tokens[:, i], film_condition) for i, head in enumerate(self.film)]
        modulated.extend((tokens[:, -1], self.language_projection(observations["language"])))
        matrix = torch.stack(modulated, 1)
        query = context + self.query_projection(detached)
        attended, _ = self.attention(query[:, None], matrix, matrix, need_weights=False)
        value = self.norm(attended[:, 0])
        return dict(conditioning=self.norm(value + self.ffn(value)), context=context, utility=utility)

    def denoising_loss(self, batch: dict, condition: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Compute per-window masked DDPM epsilon MSE using shared noise and time."""
        mask = batch["action_mask"]
        action = self.normalizer.normalize("action", batch["action"]).masked_fill(~mask, 0)
        if not mask.flatten(1).any(1).all():
            raise ValueError("Every window must have valid action targets")
        prediction = self.model(self.scheduler.add_noise(action, noise, timesteps), timesteps, global_cond=condition)
        return ((prediction - noise).square() * mask).sum((1, 2)) / mask.sum((1, 2))

    def losses(self, batch: dict, phase: str, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Keep action and SmoothL1 utility supervision independent."""
        encoded = self.encode(batch["obs"])
        utility = (nn.functional.smooth_l1_loss(encoded["utility"], batch["utility"] / self.utility_rms)
                   if self.student else encoded["context"].new_zeros(()))
        if phase == "A":
            action = utility.detach().new_zeros(())
        else:
            noise = torch.randn(batch["action"].shape, device=batch["action"].device, generator=generator)
            times = torch.randint(100, (len(noise),), device=noise.device, generator=generator)
            action = self.denoising_loss(batch, encoded["conditioning"], noise, times).mean()
        total = utility if phase == "A" else action + self.config["utility_weight"] * utility
        return dict(loss=total, action_loss=action, utility_loss=utility)

    @torch.no_grad()
    def predict_action(self, observations: dict, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        """Sample joint chunks and expose the causal executable slice."""
        condition = self.encode(observations)["conditioning"]
        action = torch.randn((len(condition), self.config["horizon"], self.config["action_dim"]),
                             device=condition.device, generator=generator)
        self.scheduler.set_timesteps(self.config["inference_steps"])
        for timestep in self.scheduler.timesteps:
            prediction = self.model(action, timestep.to(action.device), global_cond=condition)
            action = self.scheduler.step(prediction, timestep, action, generator=generator).prev_sample
        action = self.normalizer.unnormalize("action", action)
        start = self.config["action_start"]
        return dict(action_pred=action, action=action[:, start:start + self.config["n_action_steps"]])
