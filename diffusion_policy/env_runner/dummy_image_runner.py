from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class DummyImageRunner(BaseImageRunner):
    """No-op runner for offline-only training (no sim environment available)."""

    def run(self, policy):
        return {}
