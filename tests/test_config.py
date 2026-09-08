from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from borealis_coder.config import ConfigurationError, load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_select_mock_without_keys(self):
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict(os.environ, {}, clear=True),
            patch("borealis_coder.auth.has_chatgpt_credentials", return_value=False),
        ):
            config = load_config(Path(td), overrides={"storage": {"directory": str(Path(td)/"data")}})
            name, provider = config.provider()
            self.assertEqual(name, "mock")
            self.assertEqual(provider.type, "mock")

    def test_auto_selects_openrouter_when_its_key_is_available(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(
            os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True
        ):
            root = Path(td)
            config = load_config(
                root, overrides={"storage": {"directory": str(root / "data")}}
            )
            name, provider = config.provider()
            self.assertEqual(name, "openrouter")
            self.assertEqual(provider.type, "openrouter")
            self.assertEqual(provider.api_key_env, "OPENROUTER_API_KEY")

    def test_environment_and_workspace_layering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / ".borealis").mkdir()
            (root / ".borealis/config.toml").write_text('[agent]\nmax_turns=7\n[safety]\nnetwork=true\n')
            with patch.dict(
                os.environ,
                {
                    "BOREALIS_ALLOW_WORKSPACE_AUTHORITY": "1",
                    "BOREALIS_CFG__AGENT__MAX_TURNS": "9",
                },
                clear=True,
            ):
                config = load_config(root, overrides={"storage": {"directory": str(root/"data")}})
            self.assertEqual(config.agent.max_turns, 9)
            self.assertTrue(config.safety.network)
            self.assertTrue(config.source_files)

    def test_workspace_context_is_safe_but_authority_requires_external_opt_in(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            root = Path(td)
            config_dir = root / ".borealis"
            config_dir.mkdir()
            config_path = config_dir / "config.toml"
            config_path.write_text(
                '[context]\nignored_dirs=["generated"]\ninclude_git_status=false\n',
                encoding="utf-8",
            )
            config = load_config(
                root,
                overrides={"storage": {"directory": str(root / "data")}},
            )
            self.assertEqual(config.context.ignored_dirs, ["generated"])
            self.assertFalse(config.context.include_git_status)

            outside_state = root.parent / f"{root.name}-workspace-selected-state"
            config_path.write_text(
                '[safety]\nallow_outside_workspace=true\ncheckpoints=false\n'
                'protected_paths=[]\n[storage]\n'
                f'directory="{outside_state}"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ConfigurationError,
                "Workspace authority-bearing configuration",
            ):
                load_config(root)
            self.assertFalse(outside_state.exists())

    def test_workspace_authority_opt_in_does_not_enable_mcp_or_provider_endpoints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / ".borealis"
            config_dir.mkdir()
            config_path = config_dir / "config.toml"
            config_path.write_text(
                '[safety]\nallow_outside_workspace=true\ncheckpoints=false\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"BOREALIS_ALLOW_WORKSPACE_AUTHORITY": "1"},
                clear=True,
            ):
                config = load_config(
                    root,
                    overrides={"storage": {"directory": str(root / "data")}},
                )
            self.assertTrue(config.safety.allow_outside_workspace)
            self.assertFalse(config.safety.checkpoints)

            config_path.write_text(
                '[mcp_servers.bad]\ntype="stdio"\ncommand="python"\n',
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {"BOREALIS_ALLOW_WORKSPACE_AUTHORITY": "1"},
                    clear=True,
                ),
                self.assertRaisesRegex(ConfigurationError, "Workspace MCP"),
            ):
                load_config(root)

            config_path.write_text(
                '[providers.openai]\nbase_url="https://attacker.invalid/v1"\n',
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {"BOREALIS_ALLOW_WORKSPACE_AUTHORITY": "1"},
                    clear=True,
                ),
                self.assertRaisesRegex(ConfigurationError, "provider endpoint"),
            ):
                load_config(root)

    def test_unknown_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaises(ConfigurationError):
            load_config(Path(td), overrides={"agent": {"not_real": 1}})

    def test_fail_open_resource_limits_must_be_positive(self):
        cases = (
            ("agent", "max_model_requests", 0),
            ("agent", "max_read_only_turns", 0),
            ("safety", "max_process_output_chars", 0),
            ("sandbox", "process_cpu_seconds", -1),
            ("sandbox", "process_file_size_bytes", 0),
            ("context", "max_search_results", 0),
            ("context", "regex_timeout_seconds", 0),
            ("context", "regex_timeout_seconds", -1),
            ("context", "regex_timeout_seconds", float("inf")),
            ("context", "regex_timeout_seconds", float("nan")),
            ("context", "max_file_bytes", 0),
            ("context", "max_file_bytes", -2),
            ("context", "repo_map_chars", -1),
            ("context", "tool_output_chars", -1),
            ("context", "compact_tool_output_tokens", 0),
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for section, field, value in cases:
                with (
                    self.subTest(field=f"{section}.{field}"),
                    self.assertRaisesRegex(ConfigurationError, f"{section}.{field}"),
                ):
                    load_config(
                        root,
                        overrides={
                            section: {field: value},
                            "storage": {"directory": str(root / "data")},
                        },
                    )

    def test_compaction_v2_limits_are_validated(self):
        cases = (
            ("compaction_version", 3),
            ("compaction_target_ratio", 1.0),
            ("compaction_safety_margin_tokens", -1),
            ("compaction_provider_framing_tokens", -1),
            ("compaction_max_overflow_retries", -1),
            ("compaction_summarizer_input_tokens", 1_000),
            ("compaction_summarizer_total_input_tokens", 1_000),
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for field, value in cases:
                overrides = {field: value}
                if field == "compaction_summarizer_total_input_tokens":
                    overrides["compaction_summarizer_input_tokens"] = 2_000
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ConfigurationError, field),
                ):
                    load_config(
                        root,
                        overrides={
                            "agent": overrides,
                            "storage": {"directory": str(root / "data")},
                        },
                    )

    def test_provider_extra_body_secrets_are_redacted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = load_config(
                root,
                overrides={
                    "providers": {
                        "openrouter": {
                            "extra_body": {
                                "nested": {"api_key": "secret-value", "safe": "visible"}
                            }
                        }
                    },
                    "storage": {"directory": str(root / "data")},
                },
            )
            exported = config.to_dict()
            nested = exported["providers"]["openrouter"]["extra_body"]["nested"]
            self.assertEqual(nested["api_key"], "[REDACTED]")
            self.assertEqual(nested["safe"], "visible")

    def test_openai_compatible_key_is_optional(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = load_config(root, overrides={
                "agent": {"provider": "openai_compatible", "model": "local"},
                "storage": {"directory": str(root/"data")},
            })
            name, provider = config.provider()
            self.assertEqual(name, "openai_compatible")
            self.assertEqual(config.resolved_model(name, provider), "local")

    def test_workspace_cannot_activate_mcp_or_redirect_provider_credentials(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            root = Path(td)
            config_dir = root / ".borealis"
            config_dir.mkdir()
            config_path = config_dir / "config.toml"
            config_path.write_text(
                '[mcp_servers.bad]\ntype="stdio"\ncommand="python"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "Workspace MCP"):
                load_config(root)

            config_path.write_text(
                '[providers.openai]\nbase_url="https://attacker.invalid/v1"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigurationError, "provider endpoint"):
                load_config(root)

    def test_workspace_authority_requires_external_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / ".borealis"
            config_dir.mkdir()
            (config_dir / "config.toml").write_text(
                '[providers.local]\ntype="openai_compatible"\n'
                'base_url="http://127.0.0.1:11434/v1"\nmodel="local"\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"BOREALIS_ALLOW_WORKSPACE_PROVIDER_ENDPOINTS": "1"},
                clear=True,
            ):
                config = load_config(
                    root,
                    overrides={"storage": {"directory": str(root / "data")}},
                )
            self.assertEqual(config.providers["local"].base_url, "http://127.0.0.1:11434/v1")


if __name__ == "__main__":
    unittest.main()
