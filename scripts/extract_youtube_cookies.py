#!/usr/bin/env python3
"""
YouTube Cookie Extractor CLI
从浏览器提取 YouTube cookies 并保存到文件

Usage:
    python scripts/extract_youtube_cookies.py [browser]

Browser options: chrome, edge, brave, firefox (default: chrome)
"""
import sys
import os
import sqlite3
import shutil
import tempfile
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Browser cookie paths for different platforms
BROWSER_COOKIE_PATHS = {
    "darwin": {
        "chrome": "~/Library/Application Support/Google/Chrome/Default/Cookies",
        "chrome_profile": "~/Library/Application Support/Google/Chrome/{profile}/Cookies",
        "edge": "~/Library/Application Support/Microsoft Edge/Default/Cookies",
        "brave": "~/Library/Application Support/BraveSoftware/Brave-Browser/Default/Cookies",
        "opera": "~/Library/Application Support/com.operasoftware.Opera/Cookies",
        "vivaldi": "~/Library/Application Support/Vivaldi/Default/Cookies",
        "arc": "~/Library/Application Support/Arc/User Data/Default/Cookies",
    },
    "linux": {
        "chrome": "~/.config/google-chrome/Default/Cookies",
        "edge": "~/.config/microsoft-edge/Default/Cookies",
        "brave": "~/.config/BraveSoftware/Brave-Browser/Default/Cookies",
    },
    "win32": {
        "chrome": "~/AppData/Local/Google/Chrome/User Data/Default/Network/Cookies",
        "edge": "~/AppData/Local/Microsoft/Edge/User Data/Default/Network/Cookies",
        "brave": "~/AppData/Local/BraveSoftware/Brave-Browser/User Data/Default/Network/Cookies",
    },
}

YOUTUBE_DOMAINS = [".youtube.com", "youtube.com", ".google.com", ".googlevideo.com"]


def get_platform() -> str:
    if sys.platform == "darwin":
        return "darwin"
    elif sys.platform == "win32":
        return "win32"
    return "linux"


def get_chrome_profiles() -> List[str]:
    """Get all Chrome profiles"""
    platform = get_platform()
    profiles = ["Default"]

    if platform == "darwin":
        chrome_dir = Path("~/Library/Application Support/Google/Chrome").expanduser()
    elif platform == "linux":
        chrome_dir = Path("~/.config/google-chrome").expanduser()
    else:
        chrome_dir = Path("~/AppData/Local/Google/Chrome/User Data").expanduser()

    if chrome_dir.exists():
        for item in chrome_dir.iterdir():
            if item.is_dir() and item.name.startswith("Profile"):
                profiles.append(item.name)

    return profiles


def get_chrome_encryption_key() -> Optional[bytes]:
    """Get Chrome encryption key from macOS Keychain"""
    platform = get_platform()

    if platform == "darwin":
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip().encode()
        except Exception as e:
            print(f"Warning: Could not get Chrome encryption key: {e}")

    return None


def decrypt_chrome_cookie(encrypted_value: bytes, key: Optional[bytes] = None) -> Optional[str]:
    """Decrypt Chrome cookie value"""
    if not encrypted_value:
        return None

    platform = get_platform()

    # Check for v10/v11 prefix (encrypted)
    if encrypted_value[:3] in (b'v10', b'v11'):
        if platform == "darwin" and key:
            try:
                from Crypto.Cipher import AES
                from Crypto.Protocol.KDF import PBKDF2

                # Derive key using PBKDF2
                derived_key = PBKDF2(key, b'saltysalt', dkLen=16, count=1003)

                # Remove version prefix
                encrypted_value = encrypted_value[3:]

                # Decrypt using AES-CBC with 16-space IV
                iv = b' ' * 16
                cipher = AES.new(derived_key, AES.MODE_CBC, iv)
                decrypted = cipher.decrypt(encrypted_value)

                # Remove PKCS7 padding
                padding_length = decrypted[-1]
                decrypted = decrypted[:-padding_length]

                return decrypted.decode('utf-8')
            except ImportError:
                print("Warning: pycryptodome not installed. Run: pip install pycryptodome")
                return None
            except Exception as e:
                return None

        elif platform == "win32":
            try:
                import win32crypt
                decrypted = win32crypt.CryptUnprotectData(encrypted_value[3:], None, None, None, 0)[1]
                return decrypted.decode('utf-8')
            except:
                return None

    # Not encrypted or unknown format
    try:
        return encrypted_value.decode('utf-8')
    except:
        return None


def extract_cookies_from_db(db_path: Path, encryption_key: Optional[bytes] = None) -> List[Dict]:
    """Extract YouTube cookies from SQLite database"""
    cookies = []

    # Copy database to avoid lock issues
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
        tmp_path = tmp.name

    try:
        shutil.copy2(db_path, tmp_path)

        conn = sqlite3.connect(tmp_path)
        cursor = conn.cursor()

        # Query for YouTube cookies with parameterized placeholders to prevent SQL injection
        domain_placeholders = " OR ".join(["host_key LIKE ?" for _ in YOUTUBE_DOMAINS])
        domain_params = tuple(f"%{d}%" for d in YOUTUBE_DOMAINS)

        cursor.execute(f"""
            SELECT host_key, name, value, encrypted_value, path, expires_utc, is_secure, is_httponly
            FROM cookies
            WHERE {domain_placeholders}
        """, domain_params)

        for row in cursor.fetchall():
            host_key, name, value, encrypted_value, path, expires_utc, is_secure, is_httponly = row

            # Get cookie value
            cookie_value = value
            if not cookie_value and encrypted_value:
                cookie_value = decrypt_chrome_cookie(encrypted_value, encryption_key)

            if cookie_value:
                # Convert Chrome timestamp (microseconds since 1601-01-01) to Unix timestamp
                if expires_utc:
                    expires = int((expires_utc / 1000000) - 11644473600)
                else:
                    expires = 0

                cookies.append({
                    "domain": host_key,
                    "name": name,
                    "value": cookie_value,
                    "path": path or "/",
                    "expires": expires,
                    "secure": bool(is_secure),
                    "httpOnly": bool(is_httponly),
                })

        conn.close()

    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            print(f"Error: Browser database is locked. Please close {db_path.parent.parent.name} and try again.")
        else:
            print(f"Database error: {e}")
    except Exception as e:
        print(f"Error extracting cookies: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return cookies


def cookies_to_netscape(cookies: List[Dict]) -> str:
    """Convert cookies to Netscape format"""
    lines = [
        "# Netscape HTTP Cookie File",
        "# Extracted by iDubb YouTube Cookie Extractor",
        f"# Generated: {datetime.now().isoformat()}",
        ""
    ]

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


def main():
    browser = sys.argv[1] if len(sys.argv) > 1 else "chrome"
    output_path = Path(__file__).parent.parent / "data" / "youtube_cookies.txt"

    print(f"🍪 YouTube Cookie Extractor")
    print(f"=" * 50)
    print(f"Browser: {browser}")
    print(f"Output: {output_path}")
    print()

    platform = get_platform()

    # Find cookie database
    if platform not in BROWSER_COOKIE_PATHS:
        print(f"Error: Unsupported platform: {platform}")
        sys.exit(1)

    if browser not in BROWSER_COOKIE_PATHS[platform]:
        print(f"Error: Unsupported browser: {browser}")
        print(f"Available browsers: {list(BROWSER_COOKIE_PATHS[platform].keys())}")
        sys.exit(1)

    cookie_db_path = Path(BROWSER_COOKIE_PATHS[platform][browser]).expanduser()

    if not cookie_db_path.exists():
        print(f"Error: Cookie database not found: {cookie_db_path}")
        print()
        print("Tips:")
        print("1. Make sure the browser is installed")
        print("2. Make sure you've visited YouTube at least once")
        sys.exit(1)

    print(f"📂 Found cookie database: {cookie_db_path}")

    # Get encryption key for macOS Chrome
    encryption_key = None
    if platform == "darwin" and browser in ["chrome", "brave", "edge"]:
        print("🔑 Getting encryption key from Keychain...")
        encryption_key = get_chrome_encryption_key()
        if encryption_key:
            print("✅ Got encryption key")
        else:
            print("⚠️  Could not get encryption key, some cookies may not decrypt")

    # Extract cookies
    print("📥 Extracting YouTube cookies...")
    cookies = extract_cookies_from_db(cookie_db_path, encryption_key)

    if not cookies:
        print()
        print("❌ No YouTube cookies found!")
        print()
        print("Please make sure:")
        print("1. You are logged into YouTube in this browser")
        print("2. The browser is closed (to release database lock)")
        print("3. Try: pip install pycryptodome  (for encrypted cookies)")
        sys.exit(1)

    print(f"✅ Found {len(cookies)} YouTube cookies")

    # Save to file
    netscape_content = cookies_to_netscape(cookies)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(netscape_content)

    print(f"💾 Saved to: {output_path}")
    print()
    print("=" * 50)
    print("✅ Done! Now restart the backend to use new cookies:")
    print("   docker-compose restart backend")
    print()

    # Show some cookie names for verification
    cookie_names = set(c["name"] for c in cookies)
    important_cookies = ["CONSENT", "LOGIN_INFO", "SID", "HSID", "SSID", "APISID", "SAPISID"]
    found_important = [c for c in important_cookies if c in cookie_names]

    if found_important:
        print(f"🔐 Found authentication cookies: {', '.join(found_important)}")
    else:
        print("⚠️  No authentication cookies found. You may not be logged in.")


if __name__ == "__main__":
    main()
