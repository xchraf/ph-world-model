from __future__ import annotations

import unittest

import torch

from blocket_league.passive_jacobian_ph_model import (
    FrozenTransformerStateAdapter,
    UnstructuredPortConfig,
    UnstructuredPortDynamics,
    module_tensor_hash,
)
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig


class PassiveJacobianPHModelTests(unittest.TestCase):
    def _adapter(self) -> FrozenTransformerStateAdapter:
        backbone = DirectPixelTransformer(
            PixelDirectConfig(
                image_size=8,
                patch_size=4,
                palette_size=9,
                history_frames=2,
                pixel_embedding_size=4,
                hidden_size=16,
                depth=2,
                heads=4,
                mlp_ratio=2.0,
            )
        )
        return FrozenTransformerStateAdapter(backbone, 4, 12, lens_block=0)

    def test_backbone_is_frozen_but_write_has_jacobian(self) -> None:
        adapter = self._adapter()
        contexts = torch.randint(0, 9, (3, 2, 8, 8))
        mask = torch.zeros(3, 2, 4)
        mask[:, -1, 0] = 1
        write = torch.zeros(3, 16, requires_grad=True)
        state = adapter(contexts, intervention=write, intervention_mask=mask)
        gradient = torch.autograd.grad(state.square().sum(), write)[0]
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertTrue(all(not parameter.requires_grad for parameter in adapter.backbone.parameters()))

    def test_hash_ignores_adapter_updates(self) -> None:
        adapter = self._adapter()
        before = module_tensor_hash(adapter.backbone)
        with torch.no_grad():
            adapter.readout[-1].bias.add_(1.0)
        self.assertEqual(module_tensor_hash(adapter.backbone), before)

    def test_unstructured_port_is_separable_from_drift(self) -> None:
        model = UnstructuredPortDynamics(
            UnstructuredPortConfig(state_size=4, input_size=2, hidden_size=12)
        )
        state = torch.randn(5, 4)
        zero = torch.zeros(5, 2)
        torch.testing.assert_close(model.vector_field(state, zero), model.drift(state))
        self.assertEqual(model.port(state).shape, (5, 4, 2))


if __name__ == "__main__":
    unittest.main()

