"""Privacy and stable classification at the provider diagnostic boundary."""
from __future__ import annotations

import json
import unittest

from borealis_coder.agent.answers import runtime_handoff
from borealis_coder.errors import (
    ProviderAuthenticationError,
    ProviderContextOverflowError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from borealis_coder.events import Redactor
from borealis_coder.providers.diagnostics import provider_diagnostic


class ProviderDiagnosticTests(unittest.TestCase):
    def test_known_classes_status_codes_and_unknown_provider_identity(self):
        for kind, category in ((ProviderError, "provider_error"),
                               (ProviderAuthenticationError, "authentication"),
                               (ProviderContextOverflowError, "context_overflow"),
                               (ProviderRateLimitError, "rate_limit"),
                               (ProviderUnavailableError, "unavailable")):
            with self.subTest(category=category):
                diagnostic = provider_diagnostic(kind("DO_NOT_PERSIST", status_code=503,
                    retryable=True, details={"error": {"code": "rate_limit_exceeded"}}),
                    "PRIVATE_PROVIDER_NAME", Redactor())
                self.assertEqual(diagnostic, {"category": category, "status_code": 503,
                    "retryable": True, "code": "rate_limit_exceeded"})

    def test_redaction_precedes_escaping_and_rendered_diagnostic_is_bounded(self):
        secret = 'quoted"value\\nwith\ncontrols'
        redactor = Redactor([secret, "service_unavailable"])
        diagnostic = provider_diagnostic(ProviderError(secret, code="service_unavailable"),
                                         "openrouter", redactor)
        self.assertNotIn("service_unavailable", json.dumps(diagnostic))
        # Exercise the formatter independently of the fixed-size classifier.
        text = runtime_handoff(task="synthetic", cause="provider_failure", observations=[],
            changed_files=[], mutation_tracking="complete", redactor=redactor,
            provider_diagnostic={"code": secret, "other": "x" * 20_000})
        diagnostic_line = next(line for line in text.splitlines() if line.startswith("Provider diagnostic:"))
        self.assertLessEqual(len(diagnostic_line), 621)
        self.assertNotIn("quoted", text)
        self.assertIn("[REDACTED]", text)
        self.assertLessEqual(len(text), 8000)
