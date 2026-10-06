"""Text normalization for clean_reviews."""

from __future__ import annotations

import html
import re
import unicodedata

import polars as pl

from brandsentinel.core.logging import AuditLog

_HTML_TAG = re.compile(r"<[^>]+>")
_REPEATED = re.compile(r"(.)\1{3,}", flags=re.DOTALL)
_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]")
_EMOJI_NAMES = {
	"❤️": "emoji_heart",
	"👍": "emoji_thumbs_up",
	"👎": "emoji_thumbs_down",
	"🔥": "emoji_fire",
	"⚠️": "emoji_warning",
	"😊": "emoji_smile",
	"😡": "emoji_angry",
}


def _normalize_text(
	text: str | None,
	*,
	unicode_form: str,
	lowercase: bool,
	strip_html: bool,
	collapse_repeated_chars: int,
	emoji_to_token: bool,
) -> str | None:
	if text is None:
		return None
	value = unicodedata.normalize(unicode_form, text)
	if strip_html:
		value = html.unescape(_HTML_TAG.sub(" ", value))
	if emoji_to_token:
		value = _EMOJI.sub(lambda match: f" {_EMOJI_NAMES.get(match[0], 'emoji')} ", value)
	if lowercase:
		value = value.lower()
	if collapse_repeated_chars > 0:
		value = _REPEATED.sub(lambda match: match[1] * collapse_repeated_chars, value)
	return " ".join(value.split())


def normalize_reviews(
	df: pl.DataFrame,
	*,
	audit: AuditLog | None = None,
	unicode_form: str = "NFC",
	lowercase: bool = True,
	strip_html: bool = True,
	collapse_repeated_chars: int = 3,
	emoji_to_token: bool = True,
) -> pl.DataFrame:
	"""Populate text_norm while preserving raw text and recording the audit step."""
	normalized = df.with_columns(
		pl.col("text_raw")
		.map_elements(
			lambda text: _normalize_text(
				text,
				unicode_form=unicode_form,
				lowercase=lowercase,
				strip_html=strip_html,
				collapse_repeated_chars=collapse_repeated_chars,
				emoji_to_token=emoji_to_token,
			),
			return_dtype=pl.String,
		)
		.alias("text_norm")
	)
	if audit is not None:
		audit.record("normalize", df.height, normalized.height)
	return normalized
