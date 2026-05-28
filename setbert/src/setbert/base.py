"""Shared base class for the vendored DNABERT / SetBERT models.

``DbtkModel`` is a thin extension of :class:`transformers.PreTrainedModel` that
knows how to instantiate nested sub-models declared in the configuration. The
upstream deepbio-toolkit version also mixed in ``lightning.LightningModule``;
that hook is unused by ``scripts/train_setbert.py`` (which runs its own
PyTorch training loop) and by ``scripts/build_setbert_embeddings.py``, so it
has been removed to drop the PyTorch Lightning dependency.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import List, Optional, Type, TypeVar, Union

from transformers import AutoModel, PreTrainedModel, PretrainedConfig

_T = TypeVar("_T", bound=PreTrainedModel)

BaseModelType = Union[str, Path, PretrainedConfig, _T]
BaseModelClassType = Union[str, Type[_T]]


class DbtkModel(PreTrainedModel):
    """Base class for nested ``transformers`` models with sub-model wiring.

    Sub-models are listed in ``sub_models`` and resolved from the model
    configuration at construction time via :meth:`instantiate_sub_model`. Each
    sub-model contributes two config attributes:

    - ``<key>`` — either a config dict / :class:`PretrainedConfig` / model
      instance / HF repo-id string.
    - ``<key>_class`` — either a fully-qualified class path string or a class
      object.

    Example::

        class CustomConfig(PretrainedConfig):
            base: Optional[BaseModelType] = None
            base_class: Optional[BaseModelClassType] = None

        class CustomModel(DbtkModel):
            config_class = CustomConfig
            sub_models = ["base"]
    """

    config_class = PretrainedConfig

    sub_models: List[str] = []

    def __init__(self, config: Optional[Union[PretrainedConfig, dict]] = None):
        if config is None:
            config = self.config_class()
        elif isinstance(config, dict):
            config = self.config_class(**config)
        super().__init__(config)
        for model_key in self.sub_models:
            self.instantiate_sub_model(model_key)

    def instantiate_sub_model(self, model_key: str):
        """Resolve and attach the sub-model identified by ``model_key``."""
        config_key = f"{model_key}"
        class_key = f"{model_key}_class"

        model_config: Optional[BaseModelType] = getattr(self.config, config_key, None)
        model_class: Optional[BaseModelClassType] = getattr(self.config, class_key, None)
        model_instance: Optional[PreTrainedModel] = None

        if isinstance(model_class, str):
            module_name, class_name = model_class.rsplit(".", 1)
            model_class = getattr(importlib.import_module(module_name), class_name)
        model_class: Optional[Type[PreTrainedModel]] = model_class

        if isinstance(model_config, dict):
            if model_class is None:
                model_config = PretrainedConfig(**model_config)
            else:
                model_config = model_class.config_class(**model_config)

        if model_class is not None and model_config is None:
            model_config = model_class.config_class()
            model_instance = model_class(model_config)

        elif isinstance(model_config, PreTrainedModel):
            if model_class is not None and model_config.__class__ != model_class:
                raise ValueError(
                    f"Model class {model_class} does not match the class of the "
                    f"provided model {model_config.__class__}"
                )
            model_instance = model_config

        elif isinstance(model_config, PretrainedConfig):
            if model_class is None:
                model_instance = AutoModel.from_config(model_config)
            else:
                model_instance = model_class(model_config)

        elif isinstance(model_config, (str, Path)):
            if model_class is None:
                model_class = AutoModel
            if match := re.match(r"(.+\/.+)\:(.+)", model_config):
                repo, revision = match.groups()
                model_instance = model_class.from_pretrained(repo, revision=revision)
            else:
                model_instance = model_class.from_pretrained(model_config)

        if model_instance is None:
            assert model_class is None and model_config is None, (
                f"Failed to instantiate nested model: config: {model_config}, "
                f"class: {model_class}"
            )
        else:
            model_class = ".".join(
                [model_instance.__class__.__module__, model_instance.__class__.__name__]
            )
            model_config = model_instance.config

        setattr(self.config, config_key, model_config)
        setattr(self.config, class_key, model_class)
        setattr(self, model_key, model_instance)
