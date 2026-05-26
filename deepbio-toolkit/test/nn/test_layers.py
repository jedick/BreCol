from functools import partial
import numpy as np
import torch
import unittest
import unittest.mock

from dbtk.nn.layers import (
    MultiHeadAttention,
    MultiHeadAttentionBlock,
    RelativeMultiHeadAttention
)

class TestMultiHeadAttention(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factory = MultiHeadAttention

    # setup
    def setUp(self):
        torch.manual_seed(0)

    def test_io_projection_shapes(self):
        embed_dim = 64
        num_heads = 8
        layer = self.factory(embed_dim, num_heads)
        layer.eval()
        self.assertEqual(layer.w_query.in_features, embed_dim)
        self.assertEqual(layer.w_query.out_features, embed_dim)
        self.assertEqual(layer.w_key.in_features, embed_dim)
        self.assertEqual(layer.w_key.out_features, embed_dim)
        self.assertEqual(layer.w_value.in_features, embed_dim)
        self.assertEqual(layer.w_value.out_features, embed_dim)
        self.assertEqual(layer.w_output.in_features, embed_dim)
        self.assertEqual(layer.w_output.out_features, embed_dim)

    def test_io_projection_shapes_with_explicit_head_embed_dim(self):
        embed_dim = 64
        num_heads = 8
        head_dim = 64
        layer = self.factory(embed_dim, num_heads, head_embed_dim=head_dim)
        layer.eval()
        self.assertEqual(layer.w_query.in_features, embed_dim)
        self.assertEqual(layer.w_query.out_features, head_dim*num_heads)
        self.assertEqual(layer.w_key.in_features, embed_dim)
        self.assertEqual(layer.w_key.out_features, head_dim*num_heads)
        self.assertEqual(layer.w_value.in_features, embed_dim)
        self.assertEqual(layer.w_value.out_features, head_dim*num_heads)
        self.assertEqual(layer.w_output.in_features, head_dim*num_heads)
        self.assertEqual(layer.w_output.out_features, embed_dim)

    def test_merge_mask_no_masks(self):
        layer = self.factory(64, 8).eval()
        attention_mask = None
        key_padding_mask = None
        self.assertIsNone(layer.merge_mask(attention_mask, key_padding_mask))

    def test_merge_mask_with_attention_mask(self):
        layer = self.factory(64, 8).eval()
        attention_mask = torch.randint(0, 2, (2, 10, 20), dtype=torch.bool)
        key_padding_mask = None
        merged_mask = layer.merge_mask(attention_mask, key_padding_mask)
        self.assertTrue(torch.all(merged_mask == attention_mask))

    def test_merge_mask_with_key_padding_mask(self):
        layer = self.factory(64, 8).eval()
        attention_mask = None
        key_padding_mask = torch.randint(0, 2, (2, 20), dtype=torch.bool)
        merged_mask = layer.merge_mask(attention_mask, key_padding_mask)
        self.assertTrue(torch.all(merged_mask == key_padding_mask.unsqueeze(-2).expand(-1, 10, -1)))

    def test_merge_mask_with_attention_and_key_padding_mask(self):
        layer = self.factory(64, 8).eval()
        attention_mask = torch.randint(0, 2, (2, 10, 20), dtype=torch.bool)
        key_padding_mask = torch.randint(0, 2, (2, 20), dtype=torch.bool)
        merged_mask = layer.merge_mask(attention_mask, key_padding_mask)
        target = torch.logical_or(attention_mask, key_padding_mask.unsqueeze(-2).expand(-1, 10, -1))
        self.assertTrue(torch.all(merged_mask == target))

    def test_compute_attention_weights_without_masking(self):
        layer = self.factory(64, 8).eval()
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 20, 64)
        value = torch.rand(2, 20, 64)
        attention_mask = None
        key_padding_mask = None
        output, attention_weights = layer(query, key, value, key_padding_mask=key_padding_mask, attention_mask=attention_mask, return_attention_weights=True)
        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(attention_weights.shape, (2, 10, 20))

    def test_compute_attention_weights_with_attention_mask(self):
        layer = self.factory(64, 8).eval()
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 20, 64)
        value = torch.rand(2, 20, 64)
        attention_mask = torch.randint(0, 10, (2, 10, 20)) > 7
        key_padding_mask = torch.randint(0, 10, (2, 20,)) > 7
        output, attention_weights = layer(query, key, value, key_padding_mask=key_padding_mask, attention_mask=attention_mask, return_attention_weights=True)
        mask: torch.Tensor = layer.merge_mask(attention_mask, key_padding_mask) # type: ignore
        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(attention_weights.shape, (2, 10, 20))
        self.assertTrue(torch.all(attention_weights[torch.where(mask)] == 0.0))

    def test_compute_attention_weights_with_key_padding_mask(self):
        layer = self.factory(64, 8).eval()
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 20, 64)
        value = torch.rand(2, 20, 64)
        attention_mask = None
        key_padding_mask = None
        attention_head_mask = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        output, attention_weights = layer(query, key, value, key_padding_mask=key_padding_mask, attention_mask=attention_mask, attention_head_mask=attention_head_mask, average_attention_weights=False, return_attention_weights=True)
        self.assertEqual(output.shape, (2, 10, 64))
        self.assertEqual(attention_weights.shape, (2, 8, 10, 20))
        self.assertTrue(torch.all(attention_weights[:,-1,:,:] == 0.0))

    def test_backprop(self):
        layer = self.factory(64, 8)
        torch.manual_seed(1)
        query = torch.rand(2, 10, 64, requires_grad=True)
        key = torch.rand(2, 10, 64, requires_grad=True)
        value = torch.rand(2, 10, 64, requires_grad=True)
        output = layer(query, key, value)
        output.sum().backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(key.grad)
        self.assertIsNotNone(value.grad)
        self.assertFalse(torch.any(torch.isnan(query.grad)))
        self.assertFalse(torch.any(torch.isnan(key.grad)))
        self.assertFalse(torch.any(torch.isnan(value.grad)))

    def test_n_dimensional_input(self):
        layer = self.factory(64, 8)
        layer.eval()

        atol = 1e-7 # Some numerical error arises for some reason....
        torch.manual_seed(1)
        query = torch.randn(10, 64)
        key = torch.randn(10, 64)
        value = torch.randn(10, 64)
        mask = torch.randint(0, 2, (10, 10), dtype=torch.bool)
        head_mask = torch.tensor([1.0]*7 + [0.0])

        output1, attention_weights1 = layer(
            query,
            key,
            value,
            attention_mask=mask,
            attention_head_mask=head_mask,
            average_attention_weights=False,
            return_attention_weights=True)
        output2, attention_weights2 = layer(
            query.expand(2, 3, -1, -1),
            key.expand(2, 3, -1, -1),
            value.expand(2, 3, -1, -1),
            attention_mask=mask.expand(2, 3, -1, -1),
            attention_head_mask=head_mask,
            average_attention_weights=False,
            return_attention_weights=True)
        self.assertTrue(torch.all(torch.isclose(attention_weights1, attention_weights2)))
        self.assertTrue(torch.all(torch.isclose(output1.expand(2, 3, -1, -1), output2, atol=atol)))

    def test_compute_output_with_mask(self):
        layer = self.factory(64, 8).eval()
        query = torch.rand(2, 10, 64)
        key = torch.rand(2, 20, 64)
        value = torch.rand(2, 20, 64)
        key_padding_mask = torch.cat((
            torch.zeros(2, 18, dtype=torch.bool),
            torch.ones(2, 2, dtype=torch.bool),
        ), dim=-1)
        # Compute output by actually truncating the keys and values
        output_truncated = layer(query, key[:,:18,:], value[:,:18,:])
        # Compute output by masking keys and values
        output_masked = layer(query, key, value, key_padding_mask=key_padding_mask)
        self.assertTrue(torch.allclose(output_truncated, output_masked))
        self.assertFalse(torch.allclose(layer(query, key, value), output_masked))


class TestRelativeMultiHeadAttention(TestMultiHeadAttention, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factory = partial(RelativeMultiHeadAttention, max_length=15)

    def test_skew(self):
        layer = RelativeMultiHeadAttention(64, 8, max_length=5).eval()
        x = torch.arange(-4, 5).expand(5, -1)
        ans = torch.tensor([[j-i for j in range(5)] for i in range(5)])
        self.assertTrue(torch.all(layer._skew(x) == ans))


class TestMultiHeadAttentionBlock(unittest.TestCase):
    def test_n_dimensional_input(self):
        layer = MultiHeadAttentionBlock(
            MultiHeadAttention(64, 8),
            feedforward_dim=64,
            feedforward_activation=torch.nn.ReLU()
        ).eval()
        x = torch.randn(10, 64)
        y = torch.randn(20, 64)
        mask = torch.randint(0, 2, (10, 20), dtype=torch.bool)
        output1, attention_weights1 = layer(
            x,
            y,
            attention_mask=mask,
            average_attention_weights=False,
            return_attention_weights=True)
        output2, attention_weights2 = layer(
            x.expand(2, 3, -1, -1),
            y.expand(2, 3, -1, -1),
            attention_mask=mask.expand(2, 3, -1, -1),
            average_attention_weights=False,
            return_attention_weights=True)
        self.assertTrue(torch.all(torch.isclose(attention_weights1, attention_weights2)))


if __name__ == "__main__":
    unittest.main()
