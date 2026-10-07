"""
Detect and fix Zawgyi-encoded Myanmar text in pilot.csv, then normalize
everything to Unicode NFC.

Detection strategy: run a safe, unambiguous Zawgyi->Unicode character
substitution across EVERY sample and diff the result against the
original. The substitution table only contains codepoints that never
legitimately occur in correctly-encoded Unicode Myanmar text, so if a
sample changes at all, that sample contained Zawgyi. This catches both:
  - documents that are wholesale Zawgyi
  - documents that are mostly clean Unicode with a handful of stray
    Zawgyi characters mixed in (invisible to whole-document classifiers
    like myanmartools, since one odd character among thousands doesn't
    move an aggregate probability)

Conversion is then tiered by how contaminated a row is (checked via
myanmartools.ZawgyiDetector):
  - wholesale Zawgyi (detector score > 0.5): apply the fuller pipeline
    (ambiguous character reinterpretation + reordering) across the whole
    string - safe here because the whole document is Zawgyi.
  - lightly contaminated (a few stray unambiguous markers only, low
    detector score): apply ONLY the unambiguous substitution. The
    ambiguous/reordering pass is NOT safe here since it would reinterpret
    common valid Unicode characters (e.g. the asat '်') found throughout
    the otherwise-correct surrounding text.

Input:
  data_revised/mya_Mymr/annotator_batches/pilot.csv

Output:
  data_revised/mya_Mymr/annotator_batches/pilot_fixed.csv
    - same columns as input, `text` replaced with the converted/normalized
      version wherever a fix was needed.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from myanmartools import ZawgyiDetector

WHOLESALE_THRESHOLD = 0.5

ROOT = Path(__file__).parent
INPUT_PATH = ROOT / "data_revised" / "mya_Mymr" / "annotator_batches" / "pilot.csv"
OUTPUT_PATH = INPUT_PATH.with_name("pilot_fixed.csv")

# Unambiguous: these codepoints never occur in correctly-encoded Unicode
# Myanmar text, so their presence alone proves Zawgyi contamination.
ZAWGYI_ONLY_MAP = {
    'ဳ': 'ု', 'ဴ': 'ူ', 'ၚ': 'ါ်', 'ၠ': '္က', 'ၡ': '္ခ', 'ၢ': '္ဂ', 'ၣ': '္ဃ',
    'ၤ': 'င်္', 'ၥ': '္စ', 'ၦ': '္ဆ', 'ၧ': '္ဆ', 'ၨ': '္ဇ', 'ၩ': '္ဈ', 'ၪ': 'ဉ',
    'ၫ': 'ည', 'ၬ': '္ဋ', 'ၭ': '္ဌ', 'ၮ': 'ဍ္ဍ', 'ၯ': 'ဍ္ဎ', 'ၰ': '္ဏ',
    'ၱ': '္တ', 'ၲ': '္တ', 'ၳ': '္ထ', 'ၴ': '္ထ', 'ၵ': '္ဒ', 'ၶ': '္ဓ',
    'ၷ': '္ပ', 'ၸ': '္ပ', 'ၹ': '္ဖ', 'ၺ': '္ဗ', 'ၻ': '္ဘ', 'ၼ': '္မ',
    'ၽ': 'ျ', 'ၾ': 'ြ', 'ၿ': 'ြ', 'ႀ': 'ြ', 'ႁ': 'ြ', 'ႂ': 'ြ', 'ႃ': 'ြ',
    'ႄ': 'ြ', 'ႅ': '္လ', 'ႆ': 'ဿ', 'ႇ': 'ှ', 'ႈ': 'ှု', 'ႉ': 'ှူ', 'ႊ': 'ွှ',
    'ႏ': 'န', '႐': 'ရ', '႑': 'ဏ္ဍ', '႒': 'ဋ္ဌ', '႓': '္ဘ', '႔': '့', '႕': '့',
    '႗': 'ဋ္ဋ',
}
_ZAWGYI_ONLY_PATTERN = re.compile('|'.join(re.escape(k) for k in ZAWGYI_ONLY_MAP))

# Ambiguous: valid standalone Unicode characters that Zawgyi also reuses
# with different meaning. Only safe to touch once a row is already known
# (via ZAWGYI_ONLY_MAP above) to be Zawgyi text.
ZAWGYI_AMBIGUOUS_MAP = {
    '္': '်', '်': 'ျ', 'ျ': 'ြ', 'ြ': 'ွ', 'ွ': 'ှ',
    'သ္သ': 'ဿ', '၈ၤ': 'ဂင်္', 'ဧ။္': '၏', 'ဧ၊္': '၏',
    '၄င္း': '၎င်း', '၎': '၎င်း', '၎င္း': '၎င်း',
    'ေ၀': 'ေဝ', 'ေ၇': 'ေရ', 'ေ၈': 'ေဂ', 'စ်': 'ဈ', 'ဥာ': 'ဉာ',
    'ဥ္': 'ဉ်', 'ၾသ': 'ဩ', 'ေၾသာ္': 'ဪ',
}
_AMBIGUOUS_PATTERN = re.compile(
    '|'.join(re.escape(k) for k in sorted(ZAWGYI_AMBIGUOUS_MAP, key=len, reverse=True))
)

# Reordering: Zawgyi renders stacked consonants/medials in visual order;
# Unicode expects logical order.
ZAWGYI_REORDER_FIXES = [
    (re.compile(r'\s+်'), '်'),
    (re.compile(r'([က-အ])(င်္)'), r'\2\1'),
    (re.compile(r'(ေ)([က-အ၀၈၇]်[က-အ၀၈၇])'), r'\2\1'),
    (re.compile(r'([ြေ]{1,2})([က-အ၀၈၇])'), r'\2\1'),
    (re.compile(r'(ေ)([ျြွှ]+)'), r'\2\1'),
    (re.compile(r'(ှ)(ျ)'), r'\2\1'),
    (re.compile(r'(ံ)([ုူ])'), r'\2\1'),
    (re.compile(r'([ုူ])([ိီ])'), r'\2\1'),
    (re.compile(r'(ော)(်[က-အ])'), r'\2\1'),
    (re.compile(r'(ဲ)(ွ)'), r'\2\1'),
]


def spot_fix(text: str) -> str:
    """Substitute only the unambiguous Zawgyi-exclusive characters."""
    return _ZAWGYI_ONLY_PATTERN.sub(lambda m: ZAWGYI_ONLY_MAP[m.group()], text)


def full_convert(text: str) -> str:
    """Full Zawgyi->Unicode pass: unambiguous + ambiguous + reordering.
    Only call this on text already confirmed to be Zawgyi."""
    converted = spot_fix(text)
    converted = _AMBIGUOUS_PATTERN.sub(lambda m: ZAWGYI_AMBIGUOUS_MAP[m.group()], converted)
    for pattern, repl in ZAWGYI_REORDER_FIXES:
        converted = pattern.sub(repl, converted)
    return converted


def fix_row(text: str, detector: ZawgyiDetector) -> str:
    if not _ZAWGYI_ONLY_PATTERN.search(text):
        # No unambiguous Zawgyi markers at all -> genuinely clean Unicode.
        pass
    elif detector.get_zawgyi_probability(text) > WHOLESALE_THRESHOLD:
        # Whole document is Zawgyi -> safe to apply the aggressive
        # ambiguous + reordering pass across the full string.
        text = full_convert(text)
    else:
        # Mostly clean Unicode with a few stray Zawgyi characters mixed
        # in -> only touch the unambiguous characters, leave the rest of
        # the (correct) document untouched.
        text = spot_fix(text)
    return unicodedata.normalize("NFC", text)


def load_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252", sep=None, engine="python")


def main():
    detector = ZawgyiDetector()
    df = load_csv(INPUT_PATH)
    original = df["text"].astype(str)

    fixed = original.apply(lambda t: fix_row(t, detector))
    is_zawgyi = fixed != original

    print(f"unicode:     {(~is_zawgyi).sum()}")
    print(f"not unicode: {is_zawgyi.sum()}")
    if is_zawgyi.any():
        print("ids not in unicode (signal = text changed after conversion):")
        for doc_id in df.loc[is_zawgyi, "id"]:
            print(f"  {doc_id}")

    df["text"] = fixed
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
