from pathlib import Path
from transformers import PretrainedConfig, BertModel, BertConfig, PreTrainedModel, ViTModel
import unittest
from typing import Optional, Union

from dbtk.nn.models import DbtkModel

class TestSubModelConfig(PretrainedConfig):
    base: Optional[Union[str, Path, PretrainedConfig, PreTrainedModel]] = None
    base_class: Optional[str] = None

class TestSubModel(DbtkModel):
    config_class = TestSubModelConfig
    sub_models = ["base"]

class TestDbtkModel(unittest.TestCase):
    def test_init_with_no_config(self):
        """Test initialization with no config"""
        model = DbtkModel()
        self.assertIsInstance(model.config, PretrainedConfig)
        self.assertEqual(len(model.sub_models), 0)

    def test_init_with_dict_config(self):
        """Test initialization with dict config"""
        config_dict = {"hidden_size": 768}
        model = DbtkModel(config_dict)
        self.assertIsInstance(model.config, PretrainedConfig)
        self.assertEqual(model.config.hidden_size, 768)

    def test_init_with_pretrained_config(self):
        """Test initialization with PretrainedConfig"""
        config = PretrainedConfig(hidden_size=512)
        model = DbtkModel(config)
        self.assertIs(model.config, config)
        self.assertEqual(model.config.hidden_size, 512)

    def test_sub_model_instantiation_with_config_instance(self):
        """Test sub-model instantiation with config"""

        for cls in [None, BertModel, "transformers.models.bert.modeling_bert.BertModel"]:
            config = TestSubModelConfig(base=BertConfig(), base_class=cls)
            model = TestSubModel(config)
            self.assertIsInstance(model.base, BertModel)
            self.assertIsInstance(model.config.base, BertConfig)
            self.assertEqual(model.config.base_class, "transformers.models.bert.modeling_bert.BertModel")

    def test_sub_model_instantiation_with_model_instance(self):
        """Test sub-model instantiation with model instance"""

        # Test without specifying base class
        bert_model = BertModel(BertConfig())
        config = TestSubModelConfig(base=bert_model)
        model = TestSubModel(config)
        self.assertIs(model.base, bert_model)
        self.assertIs(model.config.base, bert_model.config)

        # Test with specifying base class
        config = TestSubModelConfig(base=bert_model, base_class="transformers.models.bert.modeling_bert.BertModel")
        model = TestSubModel(config)
        self.assertIs(model.base, bert_model)
        self.assertIs(model.config.base, bert_model.config)
        self.assertEqual(model.config.base_class, "transformers.models.bert.modeling_bert.BertModel")

        # Test with incorrect base class
        config = TestSubModelConfig(base=bert_model, base_class="transformers.models.vit.modeling_vit.ViTModel")
        with self.assertRaises(ValueError) as ctx:
            TestSubModel(config)

    def test_sub_model_with_pretrained_path(self):
        """Test sub-model with pretrained path"""

        for base_class in [None, "transformers.models.bert.modeling_bert.BertModel"]:
            config = TestSubModelConfig(base="bert-base-uncased", base_class=base_class)
            model = TestSubModel(config)
            self.assertIsInstance(model.base, BertModel)
            self.assertIsInstance(model.config.base, BertConfig)
            self.assertEqual(model.config.base_class, "transformers.models.bert.modeling_bert.BertModel")

    def test_sub_model_missing_config(self):
        """Test error when sub-model config is missing"""

        # Test without base model
        config = TestSubModelConfig()
        model = TestSubModel(config)
        self.assertIsNone(model.base)

        # Test with base model
        config = TestSubModelConfig(base_class="transformers.models.bert.modeling_bert.BertModel")
        model = TestSubModel(config)
        self.assertIsInstance(model.base, BertModel)
        self.assertIsInstance(model.config.base, BertConfig)

if __name__ == '__main__':
    unittest.main()
