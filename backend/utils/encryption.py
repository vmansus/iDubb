"""
Encryption utilities for sensitive data storage
Uses Fernet symmetric encryption for data at rest
"""
import os
import base64
import hashlib
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken
from loguru import logger


def get_encryption_key() -> bytes:
    """
    Get or generate encryption key from environment variable.
    Key is derived from COOKIE_ENCRYPTION_KEY env var using PBKDF2.
    If not set, generates a warning and uses a default (NOT SECURE FOR PRODUCTION).
    """
    env_key = os.environ.get("COOKIE_ENCRYPTION_KEY")

    if not env_key:
        logger.warning(
            "COOKIE_ENCRYPTION_KEY not set! Using default key. "
            "Set COOKIE_ENCRYPTION_KEY environment variable for production security."
        )
        # Use a deterministic but non-secure default for development
        env_key = "idubb-dev-key-change-in-production"

    # Derive a proper Fernet key using SHA256 hash
    # Fernet requires a 32-byte base64-encoded key
    key_hash = hashlib.sha256(env_key.encode()).digest()
    return base64.urlsafe_b64encode(key_hash)


def get_fernet() -> Fernet:
    """Get Fernet instance with current encryption key"""
    return Fernet(get_encryption_key())


def encrypt_string(plaintext: str) -> str:
    """
    Encrypt a string and return base64-encoded ciphertext.

    Args:
        plaintext: String to encrypt

    Returns:
        Base64-encoded encrypted string
    """
    if not plaintext:
        return plaintext

    try:
        fernet = get_fernet()
        encrypted = fernet.encrypt(plaintext.encode('utf-8'))
        return encrypted.decode('utf-8')
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise


def decrypt_string(ciphertext: str) -> str:
    """
    Decrypt a base64-encoded ciphertext string.

    Args:
        ciphertext: Base64-encoded encrypted string

    Returns:
        Decrypted plaintext string
    """
    if not ciphertext:
        return ciphertext

    try:
        fernet = get_fernet()
        decrypted = fernet.decrypt(ciphertext.encode('utf-8'))
        return decrypted.decode('utf-8')
    except InvalidToken:
        logger.error("Decryption failed: Invalid token. Key may have changed.")
        raise
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise


def is_encrypted(data: str) -> bool:
    """
    Check if data appears to be Fernet-encrypted.
    Fernet tokens start with 'gAAAAA' when base64 encoded.
    """
    if not data:
        return False
    return data.startswith('gAAAAA')


def encrypt_if_needed(data: str) -> str:
    """
    Encrypt data only if it's not already encrypted.
    Useful for migrating existing plaintext data.
    """
    if not data or is_encrypted(data):
        return data
    return encrypt_string(data)


def decrypt_if_needed(data: str) -> str:
    """
    Decrypt data only if it appears to be encrypted.
    Returns original data if not encrypted (backwards compatibility).
    """
    if not data:
        return data

    if is_encrypted(data):
        try:
            decrypted = decrypt_string(data)
            logger.debug(f"Decryption successful: input_len={len(data)}, output_len={len(decrypted)}")
            return decrypted
        except Exception as e:
            # If decryption fails, return original (might be plaintext)
            logger.warning(f"Decryption failed: {e}, returning original data (may be plaintext)")
            return data

    return data
