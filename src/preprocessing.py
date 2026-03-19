"""Vietnamese text preprocessing module."""

import logging
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode text to NFC form (standard Vietnamese encoding)."""
    return unicodedata.normalize("NFC", text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def clean_vietnamese_text(
    text: str,
    lowercase: bool = False,
    do_remove_urls: bool = True,
    do_remove_emails: bool = True,
    do_remove_html: bool = True,
    do_normalize_unicode: bool = True,
    do_remove_extra_whitespace: bool = True,
) -> str:
    """Clean a Vietnamese text string.

    Args:
        text: Input text.
        lowercase: Whether to lowercase the text.
        do_remove_urls: Remove URLs.
        do_remove_emails: Remove email addresses.
        do_remove_html: Remove HTML tags.
        do_normalize_unicode: Normalize to NFC unicode.
        do_remove_extra_whitespace: Collapse extra whitespace.

    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    if do_normalize_unicode:
        text = normalize_unicode(text)
    if do_remove_html:
        text = remove_html_tags(text)
    if do_remove_urls:
        text = remove_urls(text)
    if do_remove_emails:
        text = remove_emails(text)
    if lowercase:
        text = text.lower()
    if do_remove_extra_whitespace:
        text = remove_extra_whitespace(text)

    return text


def word_segment(text: str) -> str:
    """Perform Vietnamese word segmentation using underthesea."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        logger.warning("underthesea not installed. Skipping word segmentation.")
        return text


def preprocess_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply preprocessing to a DataFrame.

    Args:
        df: DataFrame with 'text' column.
        config: Preprocessing config dictionary.

    Returns:
        Preprocessed DataFrame.
    """
    preproc = config.get("preprocessing", {})

    logger.info("Cleaning text...")
    df = df.copy()
    df["text"] = df["text"].apply(
        lambda t: clean_vietnamese_text(
            t,
            lowercase=preproc.get("lowercase", False),
            do_remove_urls=preproc.get("remove_urls", True),
            do_remove_emails=preproc.get("remove_emails", True),
            do_remove_html=preproc.get("remove_html_tags", True),
            do_normalize_unicode=preproc.get("normalize_unicode", True),
            do_remove_extra_whitespace=preproc.get("remove_extra_whitespace", True),
        )
    )

    # Remove empty texts after cleaning
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    if preproc.get("word_segment", False):
        logger.info("Performing Vietnamese word segmentation...")
        df["text"] = df["text"].apply(word_segment)

    logger.info(f"Preprocessing complete. {len(df)} records remaining.")
    return df
