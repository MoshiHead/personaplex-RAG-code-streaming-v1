"""Unit tests for rag.config. Pure stdlib -- no torch/faiss/etc. required to run these."""

import os
import unittest

from rag.config import InjectionMode, RAGConfig


class TestRAGConfigDefaults(unittest.TestCase):
    def test_default_is_baseline_and_disabled(self):
        cfg = RAGConfig()
        self.assertFalse(cfg.enable_rag)
        self.assertEqual(cfg.injection_mode, InjectionMode.BASELINE)
        self.assertEqual(cfg.validate(), [])

    def test_disabled_with_non_baseline_mode_warns(self):
        cfg = RAGConfig(enable_rag=False, injection_mode=InjectionMode.PERSONA_RAG)
        warnings = cfg.validate()
        self.assertTrue(any("ENABLE_RAG is False" in w for w in warnings))

    def test_turn_injection_without_vad_warns(self):
        cfg = RAGConfig(enable_rag=True, injection_mode=InjectionMode.TURN_INJECTION, vad_enabled=False)
        warnings = cfg.validate()
        self.assertTrue(any("VAD_ENABLED=False" in w for w in warnings))

    def test_turn_injection_with_vad_is_clean(self):
        cfg = RAGConfig(enable_rag=True, injection_mode=InjectionMode.TURN_INJECTION, vad_enabled=True)
        self.assertEqual(cfg.validate(), [])

    def test_invalid_top_k_warns(self):
        cfg = RAGConfig(top_k=0)
        warnings = cfg.validate()
        self.assertTrue(any("TOP_K" in w for w in warnings))

    def test_unknown_vector_db_warns(self):
        cfg = RAGConfig(vector_db="pinecone")
        warnings = cfg.validate()
        self.assertTrue(any("VECTOR_DB" in w for w in warnings))

    def test_as_dict_serializes_enum_to_string(self):
        cfg = RAGConfig(injection_mode=InjectionMode.CACHE_AWARE)
        d = cfg.as_dict()
        self.assertEqual(d["injection_mode"], "cache_aware")
        self.assertIsInstance(d["injection_mode"], str)


class TestRAGConfigFromEnv(unittest.TestCase):
    ENV_KEYS = [
        "PERSONAPLEX_RAG_ENABLE_RAG",
        "PERSONAPLEX_RAG_INJECTION_MODE",
        "PERSONAPLEX_RAG_TOP_K",
        "PERSONAPLEX_RAG_VAD_ENABLED",
    ]

    def tearDown(self):
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)

    def test_from_env_reads_overrides(self):
        os.environ["PERSONAPLEX_RAG_ENABLE_RAG"] = "true"
        os.environ["PERSONAPLEX_RAG_INJECTION_MODE"] = "persona_rag"
        os.environ["PERSONAPLEX_RAG_TOP_K"] = "8"
        os.environ["PERSONAPLEX_RAG_VAD_ENABLED"] = "0"

        cfg = RAGConfig.from_env()

        self.assertTrue(cfg.enable_rag)
        self.assertEqual(cfg.injection_mode, InjectionMode.PERSONA_RAG)
        self.assertEqual(cfg.top_k, 8)
        self.assertFalse(cfg.vad_enabled)

    def test_from_env_defaults_when_unset(self):
        cfg = RAGConfig.from_env()
        self.assertEqual(cfg, RAGConfig())


if __name__ == "__main__":
    unittest.main()
