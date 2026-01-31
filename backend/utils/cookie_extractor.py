"""
YouTube Cookie Extractor
自动从浏览器提取 YouTube cookies
"""
import os
import sys
import json
import sqlite3
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from loguru import logger

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Browser cookie paths for different platforms
BROWSER_COOKIE_PATHS = {
    "darwin": {  # macOS
        "chrome": "~/Library/Application Support/Google/Chrome/Default/Cookies",
        "chrome_beta": "~/Library/Application Support/Google/Chrome Beta/Default/Cookies",
        "chromium": "~/Library/Application Support/Chromium/Default/Cookies",
        "edge": "~/Library/Application Support/Microsoft Edge/Default/Cookies",
        "brave": "~/Library/Application Support/BraveSoftware/Brave-Browser/Default/Cookies",
        "opera": "~/Library/Application Support/com.operasoftware.Opera/Cookies",
        "vivaldi": "~/Library/Application Support/Vivaldi/Default/Cookies",
        "firefox": "~/Library/Application Support/Firefox/Profiles",
        "safari": "~/Library/Cookies/Cookies.binarycookies",
    },
    "linux": {
        "chrome": "~/.config/google-chrome/Default/Cookies",
        "chromium": "~/.config/chromium/Default/Cookies",
        "edge": "~/.config/microsoft-edge/Default/Cookies",
        "brave": "~/.config/BraveSoftware/Brave-Browser/Default/Cookies",
        "opera": "~/.config/opera/Cookies",
        "vivaldi": "~/.config/vivaldi/Default/Cookies",
        "firefox": "~/.mozilla/firefox",
    },
    "win32": {  # Windows
        "chrome": "~/AppData/Local/Google/Chrome/User Data/Default/Network/Cookies",
        "edge": "~/AppData/Local/Microsoft/Edge/User Data/Default/Network/Cookies",
        "brave": "~/AppData/Local/BraveSoftware/Brave-Browser/User Data/Default/Network/Cookies",
        "opera": "~/AppData/Roaming/Opera Software/Opera Stable/Network/Cookies",
        "vivaldi": "~/AppData/Local/Vivaldi/User Data/Default/Network/Cookies",
        "firefox": "~/AppData/Roaming/Mozilla/Firefox/Profiles",
    },
}

# Platform domains to extract
PLATFORM_DOMAINS = {
    "youtube": [
        ".youtube.com",
        "youtube.com",
        ".google.com",
        "google.com",
        ".googlevideo.com",
    ],
    "bilibili": [
        ".bilibili.com",
        "bilibili.com",
        ".bilivideo.com",
        ".hdslb.com",
    ],
    "douyin": [
        ".douyin.com",
        "douyin.com",
        ".douyinvod.com",
        ".toutiao.com",
        ".bytedance.com",
        ".snssdk.com",
    ],
    "xiaohongshu": [
        ".xiaohongshu.com",
        "xiaohongshu.com",
        ".xhscdn.com",
    ],
}

# Required cookies for each platform
PLATFORM_REQUIRED_COOKIES = {
    "bilibili": ["SESSDATA", "bili_jct", "buvid3"],
    "douyin": ["sessionid", "sessionid_ss", "ttwid", "passport_csrf_token"],
    "xiaohongshu": ["a1", "webId", "web_session", "xsecappid"],
}

# Backward compatibility
YOUTUBE_DOMAINS = PLATFORM_DOMAINS["youtube"]


def get_platform() -> str:
    """Get current platform"""
    if sys.platform == "darwin":
        return "darwin"
    elif sys.platform == "win32":
        return "win32"
    else:
        return "linux"


def decrypt_chrome_cookie_value(encrypted_value: bytes, browser: str = "chrome") -> Optional[str]:
    """
    Decrypt Chrome cookie value.
    This is platform-specific and requires additional libraries.

    Chrome v80+ uses AES-256-GCM with v10/v11 prefix on macOS.
    """
    if not encrypted_value:
        return None

    platform = get_platform()

    try:
        if platform == "darwin":
            # macOS uses Keychain for encryption
            import subprocess

            # Get encryption key from Keychain
            safe_storage_key = None
            browser_names = {
                "chrome": "Chrome",
                "chromium": "Chromium",
                "edge": "Microsoft Edge",
                "brave": "Brave",
                "opera": "Opera",
                "vivaldi": "Vivaldi",
            }

            browser_name = browser_names.get(browser, "Chrome")

            try:
                result = subprocess.run(
                    ["security", "find-generic-password", "-w", "-s", f"{browser_name} Safe Storage"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    safe_storage_key = result.stdout.strip()
            except Exception as e:
                logger.warning(f"Failed to get keychain key: {e}")
                return None

            if not safe_storage_key:
                logger.warning(f"Could not retrieve {browser_name} Safe Storage key from Keychain")
                return None

            # Decrypt using the key
            try:
                from Crypto.Cipher import AES
                from Crypto.Protocol.KDF import PBKDF2

                # Chrome uses PBKDF2 with 1003 iterations on macOS
                key = PBKDF2(safe_storage_key.encode(), b'saltysalt', dkLen=16, count=1003)

                # Check for version prefix
                if encrypted_value[:3] == b'v10':
                    # v10 uses AES-128-CBC
                    encrypted_data = encrypted_value[3:]

                    # IV is 16 spaces for v10
                    iv = b' ' * 16
                    cipher = AES.new(key, AES.MODE_CBC, iv)
                    decrypted = cipher.decrypt(encrypted_data)

                    # Chrome on macOS adds 32 bytes of header/metadata before the actual value
                    # Skip these bytes to get the real cookie value
                    if len(decrypted) > 32:
                        decrypted = decrypted[32:]

                    # Remove PKCS7 padding
                    if decrypted:
                        padding_len = decrypted[-1]
                        if 0 < padding_len <= 16:
                            decrypted = decrypted[:-padding_len]
                        else:
                            # Invalid padding, just strip null bytes
                            decrypted = decrypted.rstrip(b'\x00')

                    # Try to decode, handling potential null terminators
                    try:
                        return decrypted.rstrip(b'\x00').decode('utf-8')
                    except UnicodeDecodeError:
                        # Try latin-1 as fallback
                        return decrypted.rstrip(b'\x00').decode('latin-1')

                elif encrypted_value[:3] == b'v11':
                    # v11 uses AES-256-GCM (Chrome 80+ on macOS)
                    from Crypto.Cipher import AES

                    # For v11, we need a 256-bit key
                    key = PBKDF2(safe_storage_key.encode(), b'saltysalt', dkLen=32, count=1003)

                    # v11: 3 byte prefix + 12 byte nonce + ciphertext + 16 byte tag
                    nonce = encrypted_value[3:15]
                    ciphertext_with_tag = encrypted_value[15:]
                    # GCM tag is last 16 bytes
                    ciphertext = ciphertext_with_tag[:-16]
                    tag = ciphertext_with_tag[-16:]

                    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                    try:
                        decrypted = cipher.decrypt_and_verify(ciphertext, tag)
                        return decrypted.decode('utf-8')
                    except Exception:
                        # Try without verification (some versions may not use proper tag)
                        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                        decrypted = cipher.decrypt(ciphertext)
                        return decrypted.rstrip(b'\x00').decode('utf-8', errors='ignore')

                else:
                    # Unknown format, try v10 approach anyway
                    iv = b' ' * 16
                    cipher = AES.new(key, AES.MODE_CBC, iv)
                    decrypted = cipher.decrypt(encrypted_value)
                    padding_len = decrypted[-1]
                    if padding_len <= 16:
                        decrypted = decrypted[:-padding_len]
                    return decrypted.rstrip(b'\x00').decode('utf-8', errors='ignore')

            except ImportError:
                logger.warning("pycryptodome not installed, trying without decryption")
                return None
            except Exception as e:
                logger.warning(f"Decryption failed: {e}")
                return None

        elif platform == "win32":
            # Windows uses DPAPI
            try:
                import win32crypt
                decrypted = win32crypt.CryptUnprotectData(encrypted_value, None, None, None, 0)[1]
                return decrypted.decode('utf-8')
            except ImportError:
                logger.warning("pywin32 not installed")
                return None
            except Exception as e:
                logger.warning(f"Windows decryption failed: {e}")
                return None

        else:
            # Linux - try without encryption first
            try:
                return encrypted_value.decode('utf-8')
            except Exception:
                return None

    except Exception as e:
        logger.error(f"Cookie decryption error: {e}")
        return None


def extract_chromium_cookies(browser: str = "chrome", target_platform: str = "youtube") -> tuple[List[Dict], str]:
    """
    Extract cookies from Chromium-based browser for a specific platform.

    Args:
        browser: Browser name (chrome, edge, brave, etc.)
        target_platform: Target platform (youtube, bilibili, douyin, xiaohongshu)

    Returns:
        Tuple of (cookies list, error message or empty string)
    """
    platform = get_platform()

    if platform not in BROWSER_COOKIE_PATHS:
        return [], f"Unsupported platform: {platform}"

    if browser not in BROWSER_COOKIE_PATHS[platform]:
        return [], f"Unsupported browser: {browser}"

    if target_platform not in PLATFORM_DOMAINS:
        return [], f"Unsupported target platform: {target_platform}"

    cookie_path = Path(BROWSER_COOKIE_PATHS[platform][browser]).expanduser()

    if not cookie_path.exists():
        return [], f"Cookie file not found at: {cookie_path}. Make sure {browser} is installed."

    cookies = []
    decryption_errors = 0
    total_platform_cookies = 0
    target_domains = PLATFORM_DOMAINS[target_platform]

    # Copy database to temp file (browser may have it locked)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        tmp_path = tmp.name

    try:
        shutil.copy2(cookie_path, tmp_path)
    except PermissionError:
        return [], f"Permission denied reading {browser} cookies. On macOS, grant 'Full Disk Access' to Terminal/your IDE in System Settings > Privacy & Security."
    except Exception as e:
        return [], f"Failed to copy cookie file: {e}"

    try:
        conn = sqlite3.connect(tmp_path)
        cursor = conn.cursor()

        # Build domain query with parameterized placeholders to prevent SQL injection
        domain_placeholders = " OR ".join(["host_key LIKE ?" for _ in target_domains])
        domain_params = tuple(f"%{d}" for d in target_domains)

        cursor.execute(f"""
            SELECT host_key, name, value, encrypted_value, path, expires_utc, is_secure, is_httponly
            FROM cookies
            WHERE {domain_placeholders}
        """, domain_params)

        rows = cursor.fetchall()
        total_platform_cookies = len(rows)

        for row in rows:
            host_key, name, value, encrypted_value, path, expires_utc, is_secure, is_httponly = row

            # Try to get value
            cookie_value = value
            if not cookie_value and encrypted_value:
                cookie_value = decrypt_chrome_cookie_value(encrypted_value, browser)
                if not cookie_value:
                    decryption_errors += 1

            if cookie_value:
                # Convert Chrome timestamp to Unix timestamp
                # Chrome uses microseconds since Jan 1, 1601
                if expires_utc:
                    expires = (expires_utc / 1000000) - 11644473600
                else:
                    expires = 0

                cookies.append({
                    "domain": host_key,
                    "name": name,
                    "value": cookie_value,
                    "path": path,
                    "expires": expires,
                    "secure": bool(is_secure),
                    "httpOnly": bool(is_httponly),
                })

        conn.close()

    except Exception as e:
        logger.error(f"Failed to extract cookies from {browser}: {e}")
        return [], f"Database error: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    platform_name = {
        "youtube": "YouTube",
        "bilibili": "Bilibili",
        "douyin": "Douyin",
        "xiaohongshu": "Xiaohongshu",
    }.get(target_platform, target_platform)

    # Generate appropriate error message
    if total_platform_cookies == 0:
        return [], f"No {platform_name} cookies found in {browser}. Make sure you're logged into {platform_name}."

    if len(cookies) == 0 and decryption_errors > 0:
        # Check if pycryptodome is installed
        try:
            from Crypto.Cipher import AES
            # Pycryptodome is installed but decryption still failed
            return [], f"Found {total_platform_cookies} cookies but failed to decrypt all of them. This may be a Keychain access issue - try running 'security find-generic-password -w -s \"Chrome Safe Storage\"' in Terminal to verify Keychain access."
        except ImportError:
            return [], "Cookie decryption requires pycryptodome. Run: pip install pycryptodome"

    return cookies, ""


def cookies_to_netscape(cookies: List[Dict]) -> str:
    """Convert cookies to Netscape format"""
    lines = ["# Netscape HTTP Cookie File", "# This file is generated by iDubb", ""]

    for cookie in cookies:
        domain = cookie["domain"]
        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = cookie.get("path", "/")
        secure = "TRUE" if cookie.get("secure") else "FALSE"
        expires = int(cookie.get("expires", 0))
        name = cookie["name"]
        value = cookie["value"]

        lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}")

    return "\n".join(lines)


def save_cookies_to_file(cookies: List[Dict], output_path: Path) -> bool:
    """Save cookies to Netscape format file"""
    try:
        netscape_content = cookies_to_netscape(cookies)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(netscape_content)

        logger.info(f"Saved {len(cookies)} cookies to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save cookies: {e}")
        return False


def extract_youtube_cookies(browser: str = "chrome", output_path: Optional[Path] = None) -> Dict:
    """
    Extract YouTube cookies from browser and save to file.

    Args:
        browser: Browser name (chrome, edge, brave, etc.)
        output_path: Output file path (default: data/youtube_cookies.txt)

    Returns:
        Dict with success status, message, and cookie count
    """
    if output_path is None:
        output_path = DATA_DIR / "youtube_cookies.txt"

    # Try to extract cookies
    cookies, error_msg = extract_chromium_cookies(browser)

    if error_msg:
        return {
            "success": False,
            "message": error_msg,
            "cookie_count": 0,
        }

    if not cookies:
        return {
            "success": False,
            "message": f"No YouTube cookies found in {browser}. Make sure you're logged into YouTube.",
            "cookie_count": 0,
        }

    # Save to file
    if save_cookies_to_file(cookies, output_path):
        return {
            "success": True,
            "message": f"Successfully extracted {len(cookies)} YouTube cookies from {browser}",
            "cookie_count": len(cookies),
            "output_path": str(output_path),
        }
    else:
        return {
            "success": False,
            "message": "Failed to save cookies to file",
            "cookie_count": len(cookies),
        }


def get_available_browsers() -> List[str]:
    """Get list of browsers with cookie files present"""
    platform = get_platform()

    if platform not in BROWSER_COOKIE_PATHS:
        return []

    available = []
    for browser, path in BROWSER_COOKIE_PATHS[platform].items():
        cookie_path = Path(path).expanduser()
        if cookie_path.exists():
            available.append(browser)

    return available


def validate_cookies(cookie_path: Path) -> Dict:
    """Validate that cookie file contains YouTube cookies with login info"""
    if not cookie_path.exists():
        return {
            "valid": False,
            "message": "Cookie file does not exist",
            "youtube_cookies": 0,
            "has_login": False,
        }

    # Key cookies required for authenticated YouTube access
    # These can be on .youtube.com OR .google.com domains
    LOGIN_COOKIES = [
        "SID", "SSID", "HSID", "APISID", "SAPISID",
        "__Secure-1PSID", "__Secure-3PSID", "LOGIN_INFO"
    ]

    # Strong login cookies that are sufficient on their own for yt-dlp
    STRONG_LOGIN_COOKIES = ["__Secure-1PSID", "__Secure-3PSID"]

    try:
        with open(cookie_path, 'r', encoding='utf-8') as f:
            content = f.read()

        youtube_count = 0
        google_count = 0
        found_login_cookies = []

        for line in content.split('\n'):
            if line.startswith('#') or not line.strip():
                continue

            parts = line.split('\t')
            if len(parts) >= 7:
                domain = parts[0]
                cookie_name = parts[5] if len(parts) > 5 else ""

                # Check if it's a YouTube domain cookie
                if ".youtube.com" in domain or domain == "youtube.com":
                    youtube_count += 1
                    # Check if it's a login cookie
                    if cookie_name in LOGIN_COOKIES and cookie_name not in found_login_cookies:
                        found_login_cookies.append(cookie_name)

                # Also check Google domain for login cookies
                if ".google.com" in domain or domain == "google.com":
                    google_count += 1
                    if cookie_name in LOGIN_COOKIES and cookie_name not in found_login_cookies:
                        found_login_cookies.append(cookie_name)

        # Has login if: 2+ login cookies OR has a strong login cookie (sufficient for yt-dlp)
        has_strong_login = any(c in STRONG_LOGIN_COOKIES for c in found_login_cookies)
        has_login = len(found_login_cookies) >= 2 or has_strong_login

        total_cookies = youtube_count + google_count
        if youtube_count == 0 and google_count == 0:
            return {
                "valid": False,
                "message": "Cookie file does not contain YouTube/Google cookies. Please export cookies from youtube.com",
                "youtube_cookies": 0,
                "google_cookies": 0,
                "has_login": False,
            }
        elif not has_login:
            return {
                "valid": False,
                "message": f"Found {youtube_count} YouTube cookies and {google_count} Google cookies but missing login cookies. Please ensure you are logged in to YouTube before exporting cookies.",
                "youtube_cookies": youtube_count,
                "google_cookies": google_count,
                "has_login": False,
                "found_login_cookies": found_login_cookies,
            }
        else:
            return {
                "valid": True,
                "message": f"Cookie file valid: {youtube_count} YouTube + {google_count} Google cookies with {len(found_login_cookies)} login cookies",
                "youtube_cookies": youtube_count,
                "google_cookies": google_count,
                "has_login": True,
                "found_login_cookies": found_login_cookies,
            }

    except Exception as e:
        return {
            "valid": False,
            "message": f"Failed to read cookie file: {e}",
            "youtube_cookies": 0,
            "google_cookies": 0,
            "has_login": False,
        }


def extract_platform_credentials(browser: str, target_platform: str) -> Dict:
    """
    Extract platform-specific credentials from browser cookies.

    Args:
        browser: Browser name (chrome, edge, brave, etc.)
        target_platform: Target platform (bilibili, douyin, xiaohongshu)

    Returns:
        Dict with success status, credentials, and message
    """
    if target_platform not in PLATFORM_REQUIRED_COOKIES:
        return {
            "success": False,
            "message": f"Unsupported platform: {target_platform}",
            "credentials": {},
        }

    # Extract cookies from browser
    cookies, error_msg = extract_chromium_cookies(browser, target_platform)

    if error_msg:
        return {
            "success": False,
            "message": error_msg,
            "credentials": {},
        }

    if not cookies:
        platform_name = {
            "bilibili": "B站",
            "douyin": "抖音",
            "xiaohongshu": "小红书",
        }.get(target_platform, target_platform)
        return {
            "success": False,
            "message": f"No {platform_name} cookies found. Please make sure you are logged in.",
            "credentials": {},
        }

    # Extract required cookies for the platform
    required_cookies = PLATFORM_REQUIRED_COOKIES[target_platform]
    found_credentials = {}

    for cookie in cookies:
        cookie_name = cookie["name"]
        if cookie_name in required_cookies:
            found_credentials[cookie_name] = cookie["value"]

    # Check if we found all required cookies
    missing_cookies = [c for c in required_cookies if c not in found_credentials]

    platform_name = {
        "bilibili": "B站",
        "douyin": "抖音",
        "xiaohongshu": "小红书",
    }.get(target_platform, target_platform)

    if target_platform == "bilibili":
        # Bilibili needs SESSDATA and bili_jct at minimum
        if "SESSDATA" in found_credentials and "bili_jct" in found_credentials:
            # Build full cookie string from all cookies (same format as other platforms)
            cookie_parts = []
            for cookie in cookies:
                cookie_parts.append(f"{cookie['name']}={cookie['value']}")
            cookie_string = "; ".join(cookie_parts)

            return {
                "success": True,
                "message": f"Successfully extracted {platform_name} credentials ({len(cookies)} cookies)",
                "credentials": {
                    "cookies": cookie_string,
                },
                "cookie_count": len(cookies),
                "found_required": list(found_credentials.keys()),
            }
        else:
            return {
                "success": False,
                "message": f"Missing required cookies for {platform_name}: {', '.join(missing_cookies)}. Please make sure you are logged in.",
                "credentials": {},
                "found": list(found_credentials.keys()),
            }

    elif target_platform in ["douyin", "xiaohongshu"]:
        # Douyin and Xiaohongshu use full cookie string
        if found_credentials:
            # Build full cookie string from all cookies
            cookie_parts = []
            for cookie in cookies:
                cookie_parts.append(f"{cookie['name']}={cookie['value']}")
            cookie_string = "; ".join(cookie_parts)

            return {
                "success": True,
                "message": f"Successfully extracted {platform_name} credentials ({len(cookies)} cookies)",
                "credentials": {
                    "cookies": cookie_string,
                },
                "cookie_count": len(cookies),
                "found_required": list(found_credentials.keys()),
            }
        else:
            return {
                "success": False,
                "message": f"Missing required cookies for {platform_name}. Please make sure you are logged in.",
                "credentials": {},
            }

    return {
        "success": False,
        "message": f"Unknown platform: {target_platform}",
        "credentials": {},
    }


def get_platform_cookie_status(target_platform: str) -> Dict:
    """
    Check if platform cookies are available in any browser.

    Returns:
        Dict with available browsers that have platform cookies
    """
    available_browsers = get_available_browsers()
    browsers_with_cookies = []

    for browser in available_browsers:
        cookies, _ = extract_chromium_cookies(browser, target_platform)
        if cookies:
            browsers_with_cookies.append(browser)

    return {
        "platform": target_platform,
        "available_browsers": available_browsers,
        "browsers_with_cookies": browsers_with_cookies,
    }


if __name__ == "__main__":
    # Test extraction
    print("Available browsers:", get_available_browsers())

    result = extract_youtube_cookies("chrome", Path("./test_cookies.txt"))
    print("YouTube extraction result:", result)

    # Test Bilibili extraction
    bilibili_result = extract_platform_credentials("chrome", "bilibili")
    print("Bilibili extraction result:", bilibili_result)
