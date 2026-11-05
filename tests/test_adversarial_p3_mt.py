"""
tests/test_adversarial_p3_mt.py
Empirical Adversarial Test Suite for Requirement R2 (Translation Quality & Vocabulary Coverage).
Evaluates:
  1. 60+ Diverse Conversational Sentences across 6 daily life domains:
     - Greetings & Pleasantries
     - Asking for directions & locations
     - Market bargaining & shopping
     - Weather expressions & nature
     - Family relationships & daily life
     - Emergency & health / disaster
  2. Numerals (simple digits to complex numerals: 7, 24, 105, 1250, 2026, 500000, 1000000)
     both standalone and embedded in conversational sentences.
  3. Compound words and hyphenated expressions (me-nyuam, poj-niam, txiv-neej, kwv-tij, etc.).
  4. Adversarial syntax stress:
     - Extreme and chaotic punctuation (?!?, ..., :::, ;)
     - Mixed and inverted casing (xIn ChÀo, HÔM NAY)
     - Irregular whitespace (tabs, newlines, multi-spaces)
     - Empty, whitespace-only, and punctuation-only inputs
     - Unicode weirdness (zero-width spaces, BOM, RTL overrides, emojis)
  5. Repetition and length stress.

Assertions enforced across ALL tests:
  - ZERO <unk> tokens emitted.
  - All translated words pass RPATextNormalizer.validate_rpa_tokens with valid 8 tones
    (-b, -m, -d, -j, -v, -s, -g, or unmarked mid tone).
  - No crash, unhandled exception, or infinite loop on adversarial syntax.
"""

import os
import sys
import unittest
from typing import List, Dict, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.mt_module import (
    HmongTranslationModel,
    RPATextNormalizer,
    HmongNumberSynthesizer
)


class TestAdversarialP3MT(unittest.TestCase):
    """
    Adversarial stress-test suite targeting Requirement R2.
    Written by challenger_p3_1 (Empirical Challenger).
    """

    @classmethod
    def setUpClass(cls):
        cls.normalizer = RPATextNormalizer()
        cls.mt = HmongTranslationModel()
        cls.valid_tones = {"b", "m", "d", "j", "v", "s", "g", ""}

    def _verify_rpa_output(self, translation: str, source_context: str = ""):
        """Helper to assert zero <unk> and 100% valid RPA 8-tone syllables."""
        # 1. Zero <unk> assertion
        self.assertNotIn(
            "<unk>",
            translation,
            f"Adversarial Failure: '<unk>' token detected in output '{translation}' for source '{source_context}'"
        )
        self.assertNotIn(
            "unk",
            translation.split(),
            f"Adversarial Failure: raw 'unk' token detected in output '{translation}' for source '{source_context}'"
        )

        # 2. Syllable and tone validity
        token_validations = self.normalizer.validate_rpa_tokens(translation)
        for token_info in token_validations:
            tok = token_info["token"]
            clean_tok = token_info["clean"]
            # Standalone digits or pure numbers are acceptable in some contexts, but let's check
            if clean_tok.isdigit():
                continue
            self.assertTrue(
                token_info["is_rpa"],
                f"Adversarial Failure: Token '{tok}' (cleaned: '{clean_tok}') in '{translation}' is NOT valid RPA! Source: '{source_context}'"
            )
            self.assertIn(
                token_info["tone"],
                self.valid_tones,
                f"Adversarial Failure: Token '{tok}' has invalid tone coda '{token_info['tone']}'! Source: '{source_context}'"
            )

    # =========================================================================
    # Test Suite 1: 60+ Conversational Sentences Across 6 Daily Life Categories
    # =========================================================================
    def test_01_conversational_sentences_across_six_categories(self):
        """
        Adversarial test on 60 conversational sentences covering:
        greetings, directions, market, weather, family, emergency/health.
        """
        sentences = [
            # 1. Greetings & Pleasantries (10 sentences)
            "Xin chào bạn",
            "Chào buổi sáng anh",
            "Chào buổi trưa chị",
            "Chào buổi tối",
            "Chúc ngủ ngon",
            "Rất vui được gặp bạn",
            "Cảm ơn bạn rất nhiều",
            "Xin lỗi bạn",
            "Không có gì đâu",
            "Tạm biệt hẹn ngày mai gặp lại",

            # 2. Asking for Directions & Locations (10 sentences)
            "Cho tôi hỏi đường đi chợ ở đâu",
            "Bệnh viện ở đâu",
            "Nhà bạn ở đâu",
            "Trường học ở chỗ nào",
            "Đi thẳng rồi rẽ phải",
            "Đi qua cầu rồi rẽ trái",
            "Gần đây có trạm y tế không",
            "Đường này đi đến thị trấn phải không",
            "Bến xe khách ở đâu",
            "Từ đây đến đó cách bao xa",

            # 3. Market Bargaining & Shopping (10 sentences)
            "Cái này giá bao nhiêu tiền",
            "Đắt quá có bớt được không",
            "Tôi muốn mua thịt lợn",
            "Bán cho tôi một cân rau cải",
            "Hai cân gạo này bao nhiêu tiền",
            "Ở đây có bán gà không",
            "Mua ba quả trứng gà",
            "Cho tôi xin tiền thừa",
            "Có mang theo tiền mặt không",
            "Tôi muốn mua một con dao quắm",

            # 4. Weather Expressions & Nature (10 sentences)
            "Hôm nay trời mưa rất to",
            "Ngày mai trời có nắng không",
            "Mùa đông ở vùng núi rất lạnh",
            "Hôm nay trời nhiều mây và có sương mù",
            "Gió thổi rất mạnh",
            "Trời sắp có giông bão",
            "Trời nắng nóng quá",
            "Mặt trời mọc ở hướng đông",
            "Mùa xuân hoa đào nở rộ",
            "Nước suối chảy xiết",

            # 5. Family Relationships & Daily Life (10 sentences)
            "Gia đình bạn có mấy người",
            "Cha mẹ tôi đang ở nhà",
            "Mẹ tôi đang nấu cơm trong bếp",
            "Anh em chúng tôi cùng đi làm nương",
            "Vợ chồng tôi có hai đứa con",
            "Ông bà nội đã lớn tuổi rồi",
            "Em gái tôi đi học ở trường",
            "Anh trai tôi đi làm ăn xa",
            "Hôm nay cả nhà cùng ăn cơm vui vẻ",
            "Bố tôi là người thợ mộc",

            # 6. Emergency, Health & Disasters (10 sentences)
            "Cứu tôi với tôi bị lạc trên núi",
            "Tôi bị đau ngực và khó thở",
            "Có sạt lở đất lớn ở bản bên cạnh",
            "Nước lũ đang dâng cao rất nguy hiểm",
            "Xin hãy gọi cấp cứu hoặc bác sĩ ngay lập tức",
            "Tôi bị sốt cao từ tối hôm qua đến giờ",
            "Bác sĩ cho tôi hỏi bệnh viện ở đâu",
            "Tôi cần mua thuốc cảm cúm và hạ sốt",
            "Bị gãy tay cần phải băng bó gấp",
            "Có người bị rắn độc cắn hãy giúp đỡ"
        ]

        self.assertGreaterEqual(len(sentences), 50, "Must test at least 50 conversational sentences")

        failures = []
        for idx, sentence in enumerate(sentences, 1):
            try:
                translation = self.mt.translate(sentence)
                self.assertIsInstance(translation, str)
                self.assertGreater(len(translation.strip()), 0, f"Empty translation for '{sentence}'")
                self._verify_rpa_output(translation, source_context=sentence)
            except AssertionError as e:
                failures.append(f"Sentence #{idx} '{sentence}' failed: {e}")

        self.assertEqual(len(failures), 0, "\n".join(failures))

    # =========================================================================
    # Test Suite 2: Numerals (Simple Digits to Complex Numerals)
    # =========================================================================
    def test_02_numerals_simple_and_complex(self):
        """
        Adversarial test on numbers: 7, 24, 105, 1250, 2026, 500000, 1000000.
        Tests:
        - Standalone integer conversion
        - Standalone Arabic digit strings
        - Standalone Vietnamese number phrases
        - Numbers embedded in conversational sentences
        """
        core_numerals = [
            (0, "zeej", "không"),
            (1, "ib", "một"),
            (7, "xya", "bảy"),
            (24, "nees nkaum plaub", "hai mươi bốn"),
            (105, "ib puas tsib", "một trăm linh năm"),
            (1250, "ib txhiab ob puas tsib caug", "một nghìn hai trăm năm mươi"),
            (2026, "ob txhiab nees nkaum rau", "hai nghìn không trăm hai mươi sáu"),
            (500000, "tsib puas txhiab", "năm trăm nghìn"),
            (1000000, "ib lab", "một triệu")
        ]

        # 1. Test HmongNumberSynthesizer.int_to_hmong_rpa directly
        for num, expected_rpa, _ in core_numerals:
            res = HmongNumberSynthesizer.int_to_hmong_rpa(num)
            self.assertEqual(res, expected_rpa, f"Number synthesizer mismatch for {num}: expected '{expected_rpa}', got '{res}'")
            self._verify_rpa_output(res, source_context=f"Synthesizer int {num}")

        # 2. Test MT translation of raw digit strings
        for num, expected_rpa, _ in core_numerals:
            digit_str = str(num)
            translation = self.mt.translate(digit_str)
            self.assertEqual(
                translation,
                expected_rpa,
                f"MT translation for digit string '{digit_str}' expected '{expected_rpa}', got '{translation}'"
            )
            self._verify_rpa_output(translation, source_context=f"MT digit '{digit_str}'")

        # 3. Test MT translation of Vietnamese number phrases
        for _, expected_rpa, vi_phrase in core_numerals:
            translation = self.mt.translate(vi_phrase)
            self.assertEqual(
                translation,
                expected_rpa,
                f"MT translation for Vi phrase '{vi_phrase}' expected '{expected_rpa}', got '{translation}'"
            )
            self._verify_rpa_output(translation, source_context=f"MT Vi phrase '{vi_phrase}'")

        # 4. Embedded numeral sentences
        num_sentences = [
            "Tôi có 7 con gà",
            "Mua 24 quả trứng gà",
            "Nhà cách đây 105 mét",
            "Giá là 1250 đồng",
            "Năm 2026",
            "Tôi có 500000 đồng"
        ]
        for s in num_sentences:
            translation = self.mt.translate(s)
            self._verify_rpa_output(translation, source_context=s)

    # =========================================================================
    # Test Suite 3: Compound Words & Hyphenated Expressions
    # =========================================================================
    def test_03_compounds_and_hyphenated_expressions(self):
        """
        Adversarial test on compound words and hyphenated expressions:
        me-nyuam, poj-niam, txiv-neej, kwv-tij, nyab-laj, hmoob-dawb, etc.
        """
        compounds = [
            ("me-nyuam", ["me", "nyuam"]),
            ("poj-niam", ["poj", "niam"]),
            ("txiv-neej", ["txiv", "neej"]),
            ("kwv-tij", ["kwv", "tij"]),
            ("nyab-laj", ["nyab", "laj"]),
            ("hmoob-dawb", ["hmoob", "dawb"])
        ]

        for comp, syllables in compounds:
            # 1. Normalizer must preserve hyphen
            norm = self.normalizer.normalize_hmong_rpa(comp)
            self.assertEqual(norm, comp, f"Normalizer damaged compound '{comp}' -> '{norm}'")

            # 2. Token validator must recognize as valid RPA compound
            val = self.normalizer.validate_rpa_tokens(comp)
            self.assertEqual(len(val), 1, f"Compound '{comp}' should parse as 1 token, got {len(val)}")
            self.assertTrue(val[0]["is_rpa"], f"Compound '{comp}' was rejected by validate_rpa_tokens")
            self.assertIn(val[0]["tone"], self.valid_tones)

        # 3. Test Vietnamese phrases that translate into compound concepts
        compound_sources = [
            "con cái",
            "vợ",
            "chồng",
            "anh em",
            "người việt nam",
            "người h'mông"
        ]
        for src in compound_sources:
            translation = self.mt.translate(src)
            self._verify_rpa_output(translation, source_context=src)

    # =========================================================================
    # Test Suite 4: Adversarial Syntax Stress (Punctuation, Casing, Whitespace)
    # =========================================================================
    def test_04_adversarial_syntax_stress(self):
        """
        Adversarial stress testing against:
        - Extreme chaotic punctuation (?!?, ..., :::, ;, [({ })])
        - Mixed, alternating, and screaming upper casing
        - Excessive tabs, newlines, and multi-spaces
        - Empty, whitespace-only, and punctuation-only edge cases
        - Unicode quirks (zero-width spaces, BOM, RTL overrides, emojis)
        """
        # A. Chaotic punctuation
        punct_inputs = [
            "Xin chào!?! ... Bạn có khỏe không??? ,,, Được rồi.",
            ";;;Chào bạn::: --- Rất vui được gặp bạn!!!",
            "((([Cảm ơn bạn]))), [Tạm biệt]...",
            "Trời mưa to! Thật không? Đúng rồi!!",
            "Bác sĩ ơi... giúp tôi với???",
            "?!?,:;--...!!!"
        ]
        for inp in punct_inputs:
            trans = self.mt.translate(inp)
            self.assertIsInstance(trans, str)
            self.assertNotIn("<unk>", trans)

        # B. Mixed and Screaming Casing
        casing_inputs = [
            "xIn ChÀo bẠn",
            "HÔM NAY TRỜI MƯA RẤT TO",
            "cẢM ƠN RẤT NHIỀU",
            "tÔI mUỐN mUA tHỊT lỢN",
            "bỆNH vIỆN Ở đÂU",
            "nEES nKAUM pLAUB"
        ]
        for inp in casing_inputs:
            trans = self.mt.translate(inp)
            self.assertIsInstance(trans, str)
            self._verify_rpa_output(trans, source_context=f"Casing: '{inp}'")

        # C. Irregular whitespace
        whitespace_inputs = [
            "   Xin     chào   \t\t   bạn   \n\n  ",
            "Hôm   nay   \r\n   trời   mưa   \t  to  ",
            "\t\t\t  cảm ơn bạn  \n\n\n",
            " \r \n \t  "
        ]
        for inp in whitespace_inputs:
            trans = self.mt.translate(inp)
            self.assertIsInstance(trans, str)
            self.assertNotIn("<unk>", trans)

        # D. Degenerate and boundary inputs
        boundary_inputs = [
            "",
            "   ",
            ".",
            "???",
            "a",
            "1"
        ]
        for inp in boundary_inputs:
            trans = self.mt.translate(inp)
            self.assertIsInstance(trans, str)
            self.assertNotIn("<unk>", trans)

        # E. Unicode quirks and emojis
        unicode_inputs = [
            "xin\u200bchào\u200cbạn",             # Zero-width space & non-joiner
            "\ufeffchào bạn",                     # Byte Order Mark
            "\u202e xin chào \u202c",             # RTL override
            "🌸 Xin chào bạn 🌼",                # Leading and trailing emojis
            "Cảm ơn bạn! 👍"                      # Emoji at end
        ]
        for inp in unicode_inputs:
            trans = self.mt.translate(inp)
            self.assertIsInstance(trans, str)
            self.assertNotIn("<unk>", trans)

    # =========================================================================
    # Test Suite 5: Repetition Stress & Token Integrity
    # =========================================================================
    def test_05_repetition_stress_and_tone_codas(self):
        """
        Adversarial test on long repeated inputs and verifying that
        all 8 tones are faithfully produced and recognized across repetitions.
        """
        repeated_text = "xin chào bạn " * 10
        trans = self.mt.translate(repeated_text, repetition_penalty=1.20)
        self.assertIsInstance(trans, str)
        self.assertNotIn("<unk>", trans)
        self.assertGreater(len(trans.strip()), 0)

        # Verify all 8 tones exist in lexicon / test outputs
        all_8_tones = {"b", "m", "d", "j", "v", "s", "g", ""}
        observed_tones = set()

        sample_words = [
            "nyob",   # -b (high level)
            "dej",    # -j (high falling)
            "kuv",    # -v (mid rising)
            "rau",    # unmarked mid
            "txias",  # -s (low level)
            "ntuj",   # -j
            "tsis",   # -s
            "pog",    # -g (low breathy)
            "kaum",   # -m (low glottalized)
            "tod"     # -d (low rising sandhi)
        ]
        for word in sample_words:
            val = self.normalizer.validate_rpa_tokens(word)
            self.assertTrue(val[0]["is_rpa"], f"Sample '{word}' should be valid RPA")
            observed_tones.add(val[0]["tone"])

        missing_tones = all_8_tones - observed_tones
        self.assertEqual(len(missing_tones), 0, f"Missing tones in verification: {missing_tones}")


if __name__ == "__main__":
    unittest.main()
