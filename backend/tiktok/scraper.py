"""
TikTok Scraper using Playwright
Fetches trending videos by tag using headless browser
"""
import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from loguru import logger

# Check if playwright is available
try:
    from playwright.async_api import async_playwright, Browser, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")


async def scrape_tiktok_tag(tag: str, max_videos: int = 20, timeout: int = 30000) -> List[Dict[str, Any]]:
    """
    Scrape TikTok videos from a tag page using Playwright.
    
    Args:
        tag: TikTok tag to scrape (without #)
        max_videos: Maximum number of videos to return
        timeout: Page load timeout in milliseconds
    
    Returns:
        List of video dictionaries
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("Playwright is not available")
        return []
    
    videos = []
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        try:
            # Create context with realistic settings
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-US',
            )
            
            page = await context.new_page()
            
            # Navigate to tag page
            url = f'https://www.tiktok.com/tag/{tag}'
            logger.info(f"Navigating to {url}")
            
            await page.goto(url, wait_until='networkidle', timeout=timeout)
            
            # Wait for video elements to load
            await page.wait_for_selector('[data-e2e="challenge-item"]', timeout=timeout)
            
            # Scroll to load more videos
            for _ in range(3):
                await page.evaluate('window.scrollBy(0, window.innerHeight)')
                await asyncio.sleep(1)
            
            # Extract video data from the page
            video_elements = await page.query_selector_all('[data-e2e="challenge-item"]')
            logger.info(f"Found {len(video_elements)} video elements")
            
            for elem in video_elements[:max_videos]:
                try:
                    video_data = await extract_video_data(elem)
                    if video_data:
                        videos.append(video_data)
                except Exception as e:
                    logger.warning(f"Failed to extract video data: {e}")
                    continue
            
            # Alternative: Try to extract from page's JSON data
            if len(videos) == 0:
                logger.info("Trying to extract from page JSON...")
                videos = await extract_from_page_json(page)
            
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
        finally:
            await browser.close()
    
    logger.info(f"Scraped {len(videos)} videos for #{tag}")
    return videos


async def extract_video_data(element) -> Optional[Dict[str, Any]]:
    """Extract video data from a video element"""
    try:
        # Get link
        link_elem = await element.query_selector('a')
        if not link_elem:
            return None
        
        href = await link_elem.get_attribute('href')
        if not href:
            return None
        
        # Extract video ID from URL
        video_id_match = re.search(r'/video/(\d+)', href)
        if not video_id_match:
            return None
        
        video_id = video_id_match.group(1)
        
        # Get thumbnail
        img_elem = await element.query_selector('img')
        thumbnail = await img_elem.get_attribute('src') if img_elem else ''
        
        # Get description/title
        desc_elem = await element.query_selector('[data-e2e="challenge-item-desc"]')
        description = await desc_elem.inner_text() if desc_elem else ''
        
        # Get author
        author_elem = await element.query_selector('[data-e2e="challenge-item-username"]')
        author = await author_elem.inner_text() if author_elem else 'Unknown'
        
        # Get stats if available
        stats_elem = await element.query_selector('[data-e2e="video-views"]')
        views_text = await stats_elem.inner_text() if stats_elem else '0'
        view_count = parse_count(views_text)
        
        return {
            'video_id': video_id,
            'title': description[:100] if description else f'TikTok Video {video_id}',
            'channel_name': author.replace('@', ''),
            'channel_url': f'https://www.tiktok.com/@{author.replace("@", "")}',
            'thumbnail_url': thumbnail,
            'video_url': f'https://www.tiktok.com{href}' if href.startswith('/') else href,
            'view_count': view_count,
            'duration': 0,  # Not easily available
            'platform': 'tiktok',
        }
    except Exception as e:
        logger.debug(f"Extract failed: {e}")
        return None


async def extract_from_page_json(page: "Page") -> List[Dict[str, Any]]:
    """Try to extract video data from embedded JSON in the page"""
    videos = []
    
    try:
        # Get page content
        content = await page.content()
        
        # Look for SIGI_STATE or similar
        patterns = [
            r'<script id="SIGI_STATE" type="application/json">(.+?)</script>',
            r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">(.+?)</script>',
            r'"ItemModule":\s*(\{.+?\})\s*,\s*"UserModule"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    # Try to find video items
                    items = find_items_in_data(data)
                    if items:
                        videos.extend(items)
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.debug(f"JSON extraction failed: {e}")
    
    return videos


def find_items_in_data(data: Any, depth: int = 0) -> List[Dict[str, Any]]:
    """Recursively search for video items in nested data"""
    if depth > 10:
        return []
    
    results = []
    
    if isinstance(data, dict):
        # Check if this looks like a video item
        if 'id' in data and ('video' in data or 'desc' in data):
            video = {
                'video_id': str(data.get('id', '')),
                'title': data.get('desc', '')[:100],
                'channel_name': data.get('author', {}).get('uniqueId', 'Unknown') if isinstance(data.get('author'), dict) else 'Unknown',
                'thumbnail_url': data.get('video', {}).get('cover', '') if isinstance(data.get('video'), dict) else '',
                'video_url': f"https://www.tiktok.com/@{data.get('author', {}).get('uniqueId', 'user')}/video/{data.get('id', '')}",
                'view_count': data.get('stats', {}).get('playCount', 0) if isinstance(data.get('stats'), dict) else 0,
                'like_count': data.get('stats', {}).get('diggCount', 0) if isinstance(data.get('stats'), dict) else 0,
                'duration': data.get('video', {}).get('duration', 0) if isinstance(data.get('video'), dict) else 0,
                'platform': 'tiktok',
            }
            if video['video_id']:
                results.append(video)
        
        # Recurse into dict values
        for key, value in data.items():
            results.extend(find_items_in_data(value, depth + 1))
    
    elif isinstance(data, list):
        for item in data:
            results.extend(find_items_in_data(item, depth + 1))
    
    return results


def parse_count(text: str) -> int:
    """Parse view/like count text like '1.2M' to integer"""
    if not text:
        return 0
    
    text = text.strip().upper()
    multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
    
    for suffix, mult in multipliers.items():
        if suffix in text:
            try:
                num = float(text.replace(suffix, '').strip())
                return int(num * mult)
            except ValueError:
                return 0
    
    try:
        return int(text.replace(',', ''))
    except ValueError:
        return 0


# Test function
async def test_scraper():
    """Test the scraper"""
    videos = await scrape_tiktok_tag('trending', max_videos=5)
    print(f"\nFound {len(videos)} videos:")
    for v in videos:
        print(f"  - {v['title'][:50]}... by {v['channel_name']}")


if __name__ == '__main__':
    asyncio.run(test_scraper())
