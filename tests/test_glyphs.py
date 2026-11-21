"""Glyph decoding on synthetic font subsets: no PDF needed.

A subset font numbers its glyphs in increasing character order, so a cipher
is built from a known string by numbering its distinct characters 1, 2, ...
and giving each glyph its Helvetica advance width.
"""
from collections import Counter

from pdfminer.fontmetrics import FONT_METRICS

from tieout import glyphs


def cipher(text: str, metrics: str = "Helvetica", fontname: str = "ABCDEF+ArialMT", size: float = 10.0):
    widths = FONT_METRICS[metrics][1]
    code = {ch: k + 1 for k, ch in enumerate(sorted(set(text), key=ord))}
    chars, x = [], 0.0
    for ch in text:
        adv = widths[ch] / 1000
        chars.append({"text": f"(cid:{code[ch]})", "fontname": fontname, "adv": adv, "size": size,
                      "x0": x, "x1": x + adv * size, "top": 100.0})
        x += adv * size
    return chars, code


def decode(text: str, vocab: Counter, judge=None):
    chars, code = cipher(text)
    out, units = glyphs.decode_pages({0: chars}, {0: {}}, vocab, judge)
    return "".join(ch["text"] for ch in out[0]), next(iter(units.values())), code


def test_width_and_order_alone_never_contradict_the_truth():
    text = "Quarterly revenue grew while operating expenses fell, and margins widened."
    _, unit, code = decode(text, Counter())
    assert unit.reference == "Helvetica"
    for ch, c in code.items():
        assert ch in unit.candidates[c]                    # the truth is always still possible
        if c in unit.decoded:
            assert unit.decoded[c][0] == ch                # and whatever is settled is right
    assert any(ev == "width" for _, ev in unit.decoded.values())


def test_crib_recovers_the_whole_string():
    text = "The total revenue was 1,234 in 2025. Net income rose to 567 from 89."
    vocab = Counter({word: 50 for word in text.split()})     # as if other filings print them often
    decoded, unit, _ = decode(text, vocab)
    assert decoded == text
    assert glyphs.summary({unit.key: unit})["undecoded"] == 0


# every printable ASCII character except the digits, which only appear as 1, 0, 2, 5
LEDGER = "".join(map(chr, range(0x20, 0x30))) + " 10 20 50 " + "".join(map(chr, range(0x3A, 0x7F)))


def digits_of(text: str) -> str:
    return "".join(ch for ch, truth in zip(text, LEDGER) if truth.isdigit())


def test_ambiguous_digits_are_reported_not_guessed():
    decoded, unit, code = decode(LEDGER, Counter())
    for d in "0125":                                     # four codes, ten equal-width digits
        assert unit.candidates[code[d]] <= set("0123456789")
        assert len(unit.candidates[code[d]]) > 1 and code[d] not in unit.decoded
    assert digits_of(decoded) == glyphs.UNKNOWN * 6
    assert all(a == b for a, b in zip(decoded, LEDGER) if a != glyphs.UNKNOWN)


def test_an_outside_judge_settles_digits():
    code = cipher(LEDGER)[1]
    truth = {code[d]: d for d in "0125"}

    def judge(units, unit, codes):
        digits = [c for c in codes if unit.candidates[c] <= set("0123456789")]
        return digits, lambda trial: sum(trial[c] == truth[c] for c in trial)

    decoded, unit, _ = decode(LEDGER, Counter(), judge)
    assert digits_of(decoded) == "102050"
    assert {unit.decoded[c][1] for c in truth} == {"arithmetic"}


def test_a_judge_that_cannot_tell_readings_apart_settles_nothing():
    text = LEDGER

    def judge(units, unit, codes):
        return codes, lambda trial: 0

    decoded, unit, _ = decode(text, Counter(), judge)
    assert glyphs.UNKNOWN in decoded
    assert not any(ev == "arithmetic" for _, ev in unit.decoded.values())


def test_monotone_choices_keep_order():
    picks = glyphs.monotone_choices([{"1", "2"}, {"1", "2", "3"}])
    assert picks == [("1", "2"), ("1", "3"), ("2", "3")]


def test_plain_text_passes_through():
    assert glyphs.translate("A", None) == "A"
    assert glyphs.cid_of("(cid:12)") == 12 and glyphs.cid_of("x") is None
