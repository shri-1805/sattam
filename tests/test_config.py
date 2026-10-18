import unittest

from human_rights_graphrag.config import Settings


class ProviderSettingsTests(unittest.TestCase):
    def test_groq_is_the_default_provider(self) -> None:
        settings = Settings(_env_file=None)

        self.assertEqual(settings.ai_provider, "groq")
        self.assertEqual(settings.llm_base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(settings.llm_model, "openai/gpt-oss-120b")
        self.assertIsNone(settings.llm_api_key)

    def test_uses_groq_credentials_for_groq_provider(self) -> None:
        settings = Settings(
            _env_file=None,
            ai_provider="groq",
            groq_api_key="test-groq-key",
            xai_api_key="unused-xai-key",
        )

        self.assertEqual(settings.llm_api_key, "test-groq-key")

    def test_tls_verification_is_enabled_by_default(self) -> None:
        settings = Settings(_env_file=None)

        self.assertTrue(settings.llm_verify_tls)

    def test_hugging_face_token_can_be_loaded_from_settings(self) -> None:
        settings = Settings(_env_file=None, hf_token="test-hf-token")

        self.assertEqual(settings.hf_token, "test-hf-token")

    def test_xai_can_remain_selected_explicitly(self) -> None:
        settings = Settings(
            _env_file=None,
            ai_provider="xai",
            xai_api_key="test-xai-key",
        )

        self.assertEqual(settings.llm_api_key, "test-xai-key")
        self.assertEqual(settings.llm_base_url, "https://api.x.ai/v1")
        self.assertEqual(settings.llm_model, "grok-4.7")


if __name__ == "__main__":
    unittest.main()
