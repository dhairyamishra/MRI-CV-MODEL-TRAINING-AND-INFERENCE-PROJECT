"""
Unit tests for configuration generation and validation.

Tests the hierarchical config system to ensure:
1. All base configs are valid YAML
2. Config merger works correctly
3. All 9 final configs can be generated
4. References are resolved properly
5. Deep merge works as expected
"""

import pytest
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.merge_configs import (
    deep_merge,
    load_yaml,
    resolve_references,
    merge_config,
    save_yaml
)


class TestConfigLoading:
    """Test that all base configs load correctly."""
    
    def test_base_configs_exist(self):
        """Test that all base config files exist."""
        base_dir = Path("configs/base")
        required_files = [
            "common.yaml",
            "model_architectures.yaml",
            "training_defaults.yaml",
            "augmentation_presets.yaml",
            "platform_overrides.yaml"
        ]
        
        for filename in required_files:
            assert (base_dir / filename).exists(), f"Missing base config: {filename}"
    
    def test_stage_configs_exist(self):
        """Test that all stage config files exist."""
        stages_dir = Path("configs/stages")
        required_files = [
            "stage1_seg_warmup.yaml",
            "stage2_cls_head.yaml",
            "stage3_joint.yaml"
        ]
        
        for filename in required_files:
            assert (stages_dir / filename).exists(), f"Missing stage config: {filename}"
    
    def test_mode_configs_exist(self):
        """Test that all mode config files exist."""
        modes_dir = Path("configs/modes")
        required_files = [
            "quick_test.yaml",
            "baseline.yaml",
            "production.yaml"
        ]
        
        for filename in required_files:
            assert (modes_dir / filename).exists(), f"Missing mode config: {filename}"
    
    def test_base_configs_valid_yaml(self):
        """Test that all base configs are valid YAML."""
        base_dir = Path("configs/base")
        for config_file in base_dir.glob("*.yaml"):
            try:
                config = load_yaml(config_file)
                assert isinstance(config, dict), f"{config_file} should be a dictionary"
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {config_file}: {e}")


class TestDeepMerge:
    """Test the deep merge functionality."""
    
    def test_simple_merge(self):
        """Test simple dictionary merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        
        assert result == {"a": 1, "b": 3, "c": 4}
    
    def test_nested_merge(self):
        """Test nested dictionary merge."""
        base = {
            "training": {
                "epochs": 100,
                "batch_size": 16,
                "early_stopping": {
                    "patience": 10
                }
            }
        }
        override = {
            "training": {
                "epochs": 50,
                "batch_size": 32
            }
        }
        result = deep_merge(base, override)
        
        assert result["training"]["epochs"] == 50
        assert result["training"]["batch_size"] == 32
        assert result["training"]["early_stopping"]["patience"] == 10
    
    def test_merge_preserves_base(self):
        """Test that merge doesn't modify the base dictionary."""
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = deep_merge(base, override)
        
        assert base == {"a": 1, "b": 2}, "Base should not be modified"
        assert result == {"a": 1, "b": 3}


class TestReferenceResolution:
    """Test reference resolution for architectures and presets."""
    
    def test_architecture_reference_resolution(self):
        """Test that architecture references are resolved correctly."""
        config = {
            "model": {
                "architecture": "multitask_medium"
            }
        }
        base_configs = {
            "architectures": {
                "multitask_medium": {
                    "in_channels": 1,
                    "base_filters": 32,
                    "depth": 4
                }
            },
            "augmentation_presets": {}
        }
        
        result = resolve_references(config, base_configs)
        
        assert "architecture" not in result["model"]
        assert result["model"]["in_channels"] == 1
        assert result["model"]["base_filters"] == 32
        assert result["model"]["depth"] == 4
    
    def test_augmentation_preset_resolution(self):
        """Test that augmentation presets are resolved correctly."""
        config = {
            "augmentation": {
                "preset": "minimal"
            }
        }
        base_configs = {
            "architectures": {},
            "augmentation_presets": {
                "minimal": {
                    "train": {
                        "enabled": True,
                        "random_flip_h": 0.5
                    }
                }
            }
        }
        
        result = resolve_references(config, base_configs)
        
        assert "preset" not in result["augmentation"]
        assert result["augmentation"]["train"]["enabled"] is True
        assert result["augmentation"]["train"]["random_flip_h"] == 0.5


class TestConfigGeneration:
    """Test full config generation for all stages and modes."""
    
    @pytest.mark.parametrize("stage", [1, 2, 3])
    @pytest.mark.parametrize("mode", ["quick", "baseline", "production"])
    def test_generate_config(self, stage, mode):
        """Test that all 9 configs can be generated without errors."""
        try:
            config = merge_config(stage, mode)
            assert isinstance(config, dict)
            assert "_metadata" in config
            assert config["_metadata"]["stage"] == stage
            assert config["_metadata"]["mode"] == mode
        except Exception as e:
            pytest.fail(f"Failed to generate stage{stage}_{mode}: {e}")
    
    def test_generated_config_has_required_keys(self):
        """Test that generated configs have all required keys."""
        config = merge_config(1, "quick")
        
        required_keys = [
            "seed", "device", "cudnn", "logging", "checkpoint",
            "training", "optimizer", "scheduler",
            "stage", "name", "model", "data", "paths", "loss"
        ]
        
        for key in required_keys:
            assert key in config, f"Missing required key: {key}"
    
    def test_stage1_quick_config(self):
        """Test specific values in stage1_quick config."""
        config = merge_config(1, "quick")
        
        assert config["stage"] == 1
        assert config["name"] == "seg_warmup"
        assert config["training"]["epochs"] == 3
        assert config["training"]["batch_size"] == 16  # User-configured batch size
        assert config["scheduler"]["T_max"] == 3
        assert config["wandb"]["enabled"] is False
    
    def test_stage3_production_config(self):
        """Test specific values in stage3_production config."""
        config = merge_config(3, "production")
        
        assert config["stage"] == 3
        assert config["name"] == "joint"
        assert config["training"]["epochs"] == 100
        assert config["training"]["batch_size"] == 32
        assert config["scheduler"]["T_max"] == 100
        assert config["wandb"]["enabled"] is True
        assert "encoder_lr" in config["optimizer"]
        assert "decoder_cls_lr" in config["optimizer"]


class TestConfigConsistency:
    """Test consistency across generated configs."""
    
    def test_all_configs_have_same_seed(self):
        """Test that all configs use the same random seed."""
        configs = [
            merge_config(stage, mode)
            for stage in [1, 2, 3]
            for mode in ["quick", "baseline", "production"]
        ]
        
        seeds = [config["seed"] for config in configs]
        assert all(seed == 42 for seed in seeds), "All configs should use seed 42"
    
    def test_all_configs_have_same_device(self):
        """Test that all configs use the same device."""
        configs = [
            merge_config(stage, mode)
            for stage in [1, 2, 3]
            for mode in ["quick", "baseline", "production"]
        ]
        
        devices = [config["device"] for config in configs]
        assert all(device == "cuda" for device in devices)
    
    def test_scheduler_tmax_matches_epochs(self):
        """Test that scheduler T_max matches training epochs."""
        for stage in [1, 2, 3]:
            for mode in ["quick", "baseline", "production"]:
                config = merge_config(stage, mode)
                assert config["scheduler"]["T_max"] == config["training"]["epochs"], \
                    f"T_max should match epochs in stage{stage}_{mode}"


class TestMetadata:
    """Test metadata tracking in generated configs."""
    
    def test_metadata_exists(self):
        """Test that metadata is added to all generated configs."""
        config = merge_config(1, "quick")
        assert "_metadata" in config
    
    def test_metadata_has_required_fields(self):
        """Test that metadata has all required fields."""
        config = merge_config(1, "quick")
        metadata = config["_metadata"]
        
        assert "stage" in metadata
        assert "mode" in metadata
        assert "generated_at" in metadata
        assert "generated_from" in metadata
    
    def test_metadata_source_files(self):
        """Test that metadata lists correct source files."""
        config = merge_config(2, "baseline")
        sources = config["_metadata"]["generated_from"]
        
        assert "base/common.yaml" in sources
        assert "base/training_defaults.yaml" in sources
        assert "stages/stage2_cls_head.yaml" in sources
        assert "modes/baseline.yaml" in sources


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
