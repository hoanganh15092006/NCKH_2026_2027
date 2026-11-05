"""
tests/verify_p3_m1_mt.py
Verification suite for Milestone P3-M1:
  1. 8 Canonical Conversational Phrases (zero <unk>, valid RPA 8 tones)
  2. Deterministic Number Synthesizer (Arabic digits & Vietnamese words)
  3. Hyphenated Compound RPA Validation (me-nyuam)
  4. 8 White Hmong RPA Tones Integrity
  5. Multi-Domain Conversational Sentences Coverage
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.mt_module import (
    HmongTranslationModel,
    RPATextNormalizer,
    HmongNumberSynthesizer
)


class TestMilestoneP3M1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mt = HmongTranslationModel()

    def test_01_canonical_conversational_phrases_zero_unk(self):
        canonical_phrases = [
            "cảm ơn bạn",
            "tạm biệt",
            "giá bao nhiêu",
            "bây giờ là mấy giờ",
            "một hai ba bốn năm",
            "hôm nay trời mưa rất to",
            "tôi muốn mua thịt lợn",
            "đường đi chợ ở đâu"
        ]
        for phrase in canonical_phrases:
            translation = self.mt.translate(phrase)
            self.assertNotIn("<unk>", translation, f"<unk> emitted in: '{phrase}' -> '{translation}'")
            val = RPATextNormalizer.validate_rpa_tokens(translation)
            self.assertTrue(len(val) > 0, f"No tokens parsed in: '{translation}'")
            for token_info in val:
                self.assertTrue(
                    token_info["is_rpa"],
                    f"Token '{token_info['token']}' is not valid RPA in translation '{translation}' of '{phrase}'"
                )

    def test_02_hmong_number_synthesizer_digits(self):
        test_cases = [
            (0, "zeej"),
            (1, "ib"),
            (5, "tsib"),
            (10, "kaum"),
            (15, "kaum tsib"),
            (20, "nees nkaum"),
            (25, "nees nkaum tsib"),
            (150, "ib puas tsib caug"),
            (2026, "ob txhiab nees nkaum rau"),
            (1000000, "ib lab")
        ]
        for num, expected in test_cases:
            res = HmongNumberSynthesizer.int_to_hmong_rpa(num)
            self.assertEqual(res, expected, f"Expected '{expected}' for {num}, got '{res}'")
            val = RPATextNormalizer.validate_rpa_tokens(res)
            self.assertTrue(all(v["is_rpa"] for v in val), f"Invalid RPA in number {num}: {res}")

    def test_03_hmong_number_synthesizer_vietnamese_text(self):
        test_cases = [
            ("mười lăm", "kaum tsib"),
            ("hai mươi", "nees nkaum"),
            ("hai mươi lăm", "nees nkaum tsib"),
            ("một trăm năm mươi", "ib puas tsib caug"),
            ("15", "kaum tsib"),
            ("25", "nees nkaum tsib"),
            ("150", "ib puas tsib caug")
        ]
        for vi_str, expected in test_cases:
            translation = self.mt.translate(vi_str)
            self.assertNotIn("<unk>", translation)
            self.assertEqual(translation, expected, f"Expected '{expected}' for '{vi_str}', got '{translation}'")

    def test_04_hyphenated_compound_validation(self):
        compounds = ["me-nyuam", "poj-niam", "txiv-neej", "kwv-tij"]
        for comp in compounds:
            val = RPATextNormalizer.validate_rpa_tokens(comp)
            self.assertEqual(len(val), 1)
            self.assertTrue(val[0]["is_rpa"], f"Expected '{comp}' to be recognized as valid RPA compound")

    def test_05_all_eight_tones_preservation(self):
        tone_samples = {
            "b": "nyob",
            "m": "kaum",
            "d": "tod",
            "j": "dej",
            "v": "kuv",
            "s": "mus",
            "g": "tsaug",
            "": "zoo"  # unmarked mid tone
        }
        for expected_tone, word in tone_samples.items():
            norm = RPATextNormalizer.normalize_hmong_rpa(f"nws pom {word} .")
            val = RPATextNormalizer.validate_rpa_tokens(norm)
            matching = [v for v in val if v["clean"] == word]
            self.assertEqual(len(matching), 1, f"Word '{word}' not found in validation output: {val}")
            actual_tone = matching[0]["tone"]
            self.assertEqual(actual_tone, expected_tone, f"Expected tone '{expected_tone}' for '{word}', got '{actual_tone}'")

    def test_06_multi_domain_conversational_coverage(self):
        sentences = [
            "chào buổi sáng", "chúc ngủ ngon", "rất vui được gặp bạn",
            "bạn có khỏe không", "tôi khỏe cảm ơn bạn", "nhà bạn ở đâu",
            "đi thẳng rồi rẽ phải", "chợ cách đây bao xa", "bây giờ tôi đói bụng",
            "tôi muốn ăn cơm", "cho tôi một cốc nước", "thịt gà giá bao nhiêu",
            "ngày mai trời có mưa không", "hôm nay trời rất nắng", "tôi bị đau bụng và sốt",
            "xin hãy gọi bác sĩ", "chúc mừng năm mới", "hẹn gặp lại ngày mai"
        ]
        for sent in sentences:
            trans = self.mt.translate(sent)
            self.assertNotIn("<unk>", trans, f"<unk> found in: '{sent}' -> '{trans}'")
            self.assertTrue(len(trans.strip()) > 0)


if __name__ == "__main__":
    unittest.main()
