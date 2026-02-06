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
    api_videos = []  # Videos captured from API responses
    
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
            
            # Intercept API responses to capture video data
            async def handle_api_response(response):
                try:
                    url = response.url
                    if 'api/challenge/item_list' in url:
                        body = await response.json()
                        if 'itemList' in body and body['itemList']:
                            for item in body['itemList']:
                                video = extract_video_from_api_item(item)
                                if video:
                                    api_videos.append(video)
                            logger.info(f"Captured {len(body['itemList'])} videos from API response")
                except Exception as e:
                    logger.debug(f"Failed to parse API response: {e}")
            
            page.on('response', handle_api_response)
            
            # Navigate to tag page
            url = f'https://www.tiktok.com/tag/{tag}'
            logger.info(f"Navigating to {url}")
            
            await page.goto(url, wait_until='networkidle', timeout=timeout)
            
            # Wait for page to load
            try:
                await page.wait_for_selector('[data-e2e="challenge-item"]', timeout=timeout)
            except:
                logger.debug("Challenge item selector not found, continuing...")
            
            # Wait a bit for API calls to complete
            await asyncio.sleep(2)
            
            # Scroll to load more videos (triggers more API calls)
            for _ in range(3):
                await page.evaluate('window.scrollBy(0, window.innerHeight)')
                await asyncio.sleep(1)
            
            # Use API-captured videos if available (preferred - has full stats)
            if api_videos:
                logger.info(f"Got {len(api_videos)} videos from API interception with full stats")
                videos = api_videos
            else:
                # Fallback: Try JSON extraction from page
                logger.info("No API videos captured, trying page JSON extraction...")
                videos = await extract_from_page_json(page)
            
            # If still no videos with stats, fallback to DOM extraction
            if not videos or not any(v.get('view_count', 0) > 0 for v in videos):
                # Fallback to DOM extraction
                logger.info("JSON extraction incomplete, trying DOM extraction...")
                video_elements = await page.query_selector_all('[data-e2e="challenge-item"]')
                logger.info(f"Found {len(video_elements)} video elements")
                
                dom_videos = []
                for elem in video_elements[:max_videos]:
                    try:
                        video_data = await extract_video_data(elem)
                        if video_data:
                            dom_videos.append(video_data)
                    except Exception as e:
                        logger.warning(f"Failed to extract video data: {e}")
                        continue
                
                # Merge: prefer API/JSON data but add DOM videos if they have more info
                if dom_videos:
                    if not videos:
                        videos = dom_videos
                    else:
                        # Merge by video_id, preferring data with stats
                        video_map = {v['video_id']: v for v in videos}
                        for dv in dom_videos:
                            vid = dv['video_id']
                            if vid not in video_map:
                                video_map[vid] = dv
                            elif dv.get('view_count', 0) > video_map[vid].get('view_count', 0):
                                video_map[vid].update(dv)
                        videos = list(video_map.values())
            
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
        
        # Get description/title - try multiple selectors
        description = ''
        for selector in ['[data-e2e="challenge-item-desc"]', '[class*="DivVideoTitle"]', '[class*="title"]', 'a[title]']:
            desc_elem = await element.query_selector(selector)
            if desc_elem:
                if selector == 'a[title]':
                    description = await desc_elem.get_attribute('title') or ''
                else:
                    description = await desc_elem.inner_text()
                if description:
                    break
        
        # Get author - try multiple selectors
        author = 'Unknown'
        for selector in ['[data-e2e="challenge-item-username"]', '[class*="AuthorTitle"]', '[class*="author"]', 'a[href*="/@"]']:
            author_elem = await element.query_selector(selector)
            if author_elem:
                author = await author_elem.inner_text()
                if not author and selector == 'a[href*="/@"]':
                    author_href = await author_elem.get_attribute('href')
                    if author_href:
                        author_match = re.search(r'/@([^/]+)', author_href)
                        if author_match:
                            author = author_match.group(1)
                if author and author != 'Unknown':
                    break
        
        # Get stats - try multiple selectors for views
        view_count = 0
        for selector in ['[data-e2e="video-views"]', '[class*="video-count"]', '[class*="PlayCount"]', 'strong[data-e2e]']:
            stats_elem = await element.query_selector(selector)
            if stats_elem:
                views_text = await stats_elem.inner_text()
                view_count = parse_count(views_text)
                if view_count > 0:
                    break
        
        # Get duration - look for time display
        duration = 0
        for selector in ['[class*="DivTimeTag"]', '[class*="duration"]', '[class*="time"]']:
            time_elem = await element.query_selector(selector)
            if time_elem:
                time_text = await time_elem.inner_text()
                duration = parse_duration(time_text)
                if duration > 0:
                    break
        
        return {
            'video_id': video_id,
            'title': description[:100] if description else f'TikTok Video {video_id}',
            'channel_name': author.replace('@', ''),
            'channel_url': f'https://www.tiktok.com/@{author.replace("@", "")}',
            'thumbnail_url': thumbnail,
            'video_url': f'https://www.tiktok.com{href}' if href.startswith('/') else href,
            'view_count': view_count,
            'duration': duration,
            'platform': 'tiktok',
        }
    except Exception as e:
        logger.debug(f"Extract failed: {e}")
        return None


def extract_video_from_api_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract video data from TikTok API response item"""
    from datetime import datetime, timezone
    
    try:
        video_id = str(item.get('id', ''))
        if not video_id:
            return None
        
        # Get author info
        author = item.get('author', {})
        author_id = author.get('uniqueId', 'unknown')
        
        # Get stats
        stats = item.get('stats', {})
        view_count = stats.get('playCount', 0)
        
        # Get video info
        video_info = item.get('video', {})
        duration = video_info.get('duration', 0)
        cover = video_info.get('cover', '') or video_info.get('originCover', '')
        
        # Get description
        desc = item.get('desc', '')
        
        # Get publish time (createTime is Unix timestamp)
        # Convert to user's configured timezone
        create_time = item.get('createTime', 0)
        published_at = None
        if create_time:
            try:
                from settings_store import settings_store
                import pytz
                
                # Get timezone from processing settings
                tz_name = settings_store.load().processing.timezone or 'Asia/Shanghai'
                tz = pytz.timezone(tz_name)
                
                # Convert Unix timestamp to timezone-aware datetime
                utc_dt = datetime.fromtimestamp(int(create_time), tz=timezone.utc)
                local_dt = utc_dt.astimezone(tz)
                published_at = local_dt.isoformat()
            except Exception as e:
                # Fallback to UTC if timezone conversion fails
                logger.debug(f"Timezone conversion failed: {e}, using UTC")
                published_at = datetime.utcfromtimestamp(int(create_time)).isoformat()
        
        return {
            'video_id': video_id,
            'title': desc[:100] if desc else f'TikTok Video {video_id}',
            'channel_name': author_id,
            'channel_url': f'https://www.tiktok.com/@{author_id}',
            'thumbnail_url': cover,
            'video_url': f'https://www.tiktok.com/@{author_id}/video/{video_id}',
            'view_count': view_count,
            'duration': duration,
            'platform': 'tiktok',
            'published_at': published_at,
        }
    except Exception as e:
        logger.debug(f"Failed to extract from API item: {e}")
        return None


def parse_duration(text: str) -> int:
    """Parse duration text like '1:30' or '01:30' to seconds"""
    if not text:
        return 0
    
    text = text.strip()
    
    # Try MM:SS format
    match = re.match(r'(\d+):(\d+)', text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes * 60 + seconds
    
    # Try just seconds
    try:
        return int(text)
    except ValueError:
        return 0


async def extract_from_page_json(page: "Page") -> List[Dict[str, Any]]:
    """Try to extract video data from embedded JSON in the page"""
    videos = []
    
    try:
        # Method 1: Use JavaScript to access window.__NEXT_DATA__ or similar
        json_data = await page.evaluate('''() => {
            // Try different global data sources
            if (window.__NEXT_DATA__) return window.__NEXT_DATA__;
            if (window.SIGI_STATE) return window.SIGI_STATE;
            if (window.__UNIVERSAL_DATA_FOR_REHYDRATION__) return window.__UNIVERSAL_DATA_FOR_REHYDRATION__;
            
            // Try to find script tags with JSON data
            const scripts = document.querySelectorAll('script[type="application/json"]');
            for (const script of scripts) {
                try {
                    const data = JSON.parse(script.textContent);
                    if (data && (data.ItemModule || data.props || data.__DEFAULT_SCOPE__)) {
                        return data;
                    }
                } catch (e) {}
            }
            return null;
        }''')
        
        if json_data:
            logger.debug(f"Found JSON data with keys: {list(json_data.keys())[:5] if isinstance(json_data, dict) else 'not dict'}")
            items = find_items_in_data(json_data)
            if items:
                videos.extend(items)
                logger.info(f"Extracted {len(items)} videos from page JSON")
                return videos
        
        # Method 2: Parse from page content (fallback)
        content = await page.content()
        
        patterns = [
            r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.+?)</script>',
            r'<script id="SIGI_STATE"[^>]*>(.+?)</script>',
            r'<script id="__NEXT_DATA__"[^>]*>(.+?)</script>',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    logger.debug(f"Found JSON via regex, keys: {list(data.keys())[:5] if isinstance(data, dict) else 'not dict'}")
                    items = find_items_in_data(data)
                    if items:
                        videos.extend(items)
                        logger.info(f"Extracted {len(items)} videos from page JSON (regex)")
                        break
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode error: {e}")
                    continue
    except Exception as e:
        logger.debug(f"JSON extraction failed: {e}")
    
    return videos


def find_items_in_data(data: Any, depth: int = 0) -> List[Dict[str, Any]]:
    """Recursively search for video items in nested data"""
    if depth > 15:
        return []
    
    results = []
    
    if isinstance(data, dict):
        # Log top-level keys for debugging
        if depth == 0:
            logger.debug(f"Top-level keys in data: {list(data.keys())[:10]}")
        
        # Check for ItemModule (common in TikTok's data structure)
        if 'ItemModule' in data and isinstance(data['ItemModule'], dict):
            logger.info(f"Found ItemModule with {len(data['ItemModule'])} items")
            for video_id, item_data in data['ItemModule'].items():
                video = extract_video_from_item(item_data, video_id)
                if video:
                    results.append(video)
            if results:
                return results  # Return early if we found good data
        
        # Check for __DEFAULT_SCOPE__ structure (newer TikTok pages)
        if '__DEFAULT_SCOPE__' in data:
            scope = data['__DEFAULT_SCOPE__']
            logger.debug(f"Found __DEFAULT_SCOPE__ with keys: {list(scope.keys())[:10] if isinstance(scope, dict) else 'not dict'}")
            if isinstance(scope, dict):
                # Look for specific keys that contain video data
                video_keys = [k for k in scope.keys() if any(x in k.lower() for x in ['item', 'video', 'feed', 'challenge'])]
                for key in video_keys:
                    logger.debug(f"Checking __DEFAULT_SCOPE__.{key}")
                    results.extend(find_items_in_data(scope[key], depth + 1))
        
        # Check for props.pageProps structure (Next.js pattern)
        if 'props' in data and isinstance(data['props'], dict):
            page_props = data['props'].get('pageProps', {})
            if page_props:
                logger.debug(f"Found props.pageProps with keys: {list(page_props.keys())[:10]}")
                results.extend(find_items_in_data(page_props, depth + 1))
        
        # Check if this looks like a video item
        if 'id' in data and ('video' in data or 'desc' in data or 'stats' in data):
            video = extract_video_from_item(data)
            if video:
                results.append(video)
        
        # Check for itemList or similar arrays
        array_keys = ['itemList', 'items', 'videoList', 'videos', 'aweme_list', 'ItemList', 'challengeItem', 'itemInfos']
        for key in array_keys:
            if key in data and isinstance(data[key], list):
                logger.debug(f"Found {key} with {len(data[key])} items")
                for item in data[key]:
                    video = extract_video_from_item(item)
                    if video:
                        results.append(video)
        
        # Recurse into dict values (but skip already processed keys)
        skip_keys = {'ItemModule', '__DEFAULT_SCOPE__', 'props', 'itemList', 'items', 'videoList', 'videos', 'aweme_list', 'ItemList', 'challengeItem', 'itemInfos'}
        for key, value in data.items():
            if key not in skip_keys:
                results.extend(find_items_in_data(value, depth + 1))
    
    elif isinstance(data, list):
        for item in data:
            results.extend(find_items_in_data(item, depth + 1))
    
    # Remove duplicates by video_id
    seen = set()
    unique_results = []
    for video in results:
        vid = video.get('video_id')
        if vid and vid not in seen:
            seen.add(vid)
            unique_results.append(video)
    
    # Sort by view_count (highest first) and then by create_time (newest first)
    unique_results.sort(key=lambda x: (x.get('view_count', 0), x.get('create_time', 0)), reverse=True)
    
    return unique_results


def extract_video_from_item(data: Dict, video_id: str = None) -> Optional[Dict[str, Any]]:
    """Extract video info from a TikTok item data structure"""
    if not isinstance(data, dict):
        return None
    
    vid = video_id or str(data.get('id', ''))
    if not vid:
        return None
    
    # Get author info - try multiple paths
    author = data.get('author', {})
    if isinstance(author, dict):
        author_name = author.get('uniqueId') or author.get('nickname') or author.get('unique_id') or 'Unknown'
    else:
        author_name = data.get('authorUniqueId', '') or data.get('author', 'Unknown')
    
    # Get stats - try multiple paths and field names
    stats = data.get('stats', {}) or data.get('statistics', {}) or data.get('statsV2', {})
    view_count = 0
    like_count = 0
    
    if isinstance(stats, dict):
        # Try all possible field names for play count
        view_count = (
            stats.get('playCount') or 
            stats.get('play_count') or 
            stats.get('viewCount') or 
            stats.get('playCountStr') or  # Sometimes it's a string like "1.2M"
            0
        )
        if isinstance(view_count, str):
            view_count = parse_count(view_count)
        
        like_count = (
            stats.get('diggCount') or 
            stats.get('digg_count') or 
            stats.get('likeCount') or 
            stats.get('likes') or
            0
        )
        if isinstance(like_count, str):
            like_count = parse_count(like_count)
    
    # Also check top-level fields
    if not view_count:
        view_count = data.get('playCount') or data.get('play_count') or 0
        if isinstance(view_count, str):
            view_count = parse_count(view_count)
    
    # Get video info
    video_info = data.get('video', {})
    duration = 0
    thumbnail = ''
    
    if isinstance(video_info, dict):
        duration = video_info.get('duration', 0)
        thumbnail = (
            video_info.get('cover') or 
            video_info.get('dynamicCover') or 
            video_info.get('originCover') or 
            video_info.get('playAddr') or
            ''
        )
    
    # Fallback for duration
    if not duration:
        duration = data.get('duration', 0) or data.get('videoDuration', 0)
    
    # Get description/title
    desc = data.get('desc', '') or data.get('title', '') or data.get('description', '')
    
    # Get createTime for sorting by recency
    create_time = data.get('createTime', 0) or data.get('create_time', 0)
    
    video_result = {
        'video_id': vid,
        'title': desc[:100] if desc else f'TikTok Video {vid}',
        'channel_name': str(author_name).replace('@', ''),
        'thumbnail_url': thumbnail,
        'video_url': f"https://www.tiktok.com/@{author_name}/video/{vid}",
        'view_count': int(view_count) if view_count else 0,
        'like_count': int(like_count) if like_count else 0,
        'duration': int(duration) if duration else 0,
        'platform': 'tiktok',
        'create_time': int(create_time) if create_time else 0,
    }
    
    # Debug logging for first few videos
    logger.debug(f"Extracted video {vid}: views={view_count}, dur={duration}, title={desc[:30] if desc else 'N/A'}")
    
    return video_result


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


async def scrape_tiktok_discover(max_videos: int = 20, timeout: int = 30000) -> List[Dict[str, Any]]:
    """
    Scrape TikTok Explore/Discover page for trending videos.
    This page typically shows more popular and recent content.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("Playwright is not available")
        return []
    
    videos = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        try:
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='en-US',
            )
            
            page = await context.new_page()
            
            # Try different URLs for discover/trending content
            urls_to_try = [
                'https://www.tiktok.com/explore',
                'https://www.tiktok.com/foryou',
                'https://www.tiktok.com/trending',
            ]
            
            for url in urls_to_try:
                try:
                    logger.info(f"Trying to scrape {url}")
                    await page.goto(url, wait_until='networkidle', timeout=timeout)
                    await asyncio.sleep(2)
                    
                    # First try to extract from page JSON (most reliable for stats)
                    videos = await extract_from_page_json(page)
                    if videos:
                        logger.info(f"Extracted {len(videos)} videos from JSON at {url}")
                        break
                    
                except Exception as e:
                    logger.debug(f"Failed to scrape {url}: {e}")
                    continue
            
            # If JSON extraction failed, try DOM scraping
            if not videos:
                logger.info("JSON extraction failed, trying DOM scraping...")
                
                # Scroll to load more content
                for _ in range(5):
                    await page.evaluate('window.scrollBy(0, window.innerHeight)')
                    await asyncio.sleep(1)
                
                # Try various video container selectors
                selectors = [
                    '[data-e2e="recommend-list-item-container"]',
                    '[class*="DivItemContainerV2"]',
                    '[class*="video-feed-item"]',
                    'div[class*="tiktok-"][data-index]',
                ]
                
                for selector in selectors:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        logger.info(f"Found {len(elements)} elements with selector: {selector}")
                        for elem in elements[:max_videos]:
                            try:
                                video_data = await extract_video_data(elem)
                                if video_data:
                                    videos.append(video_data)
                            except Exception as e:
                                logger.debug(f"Failed to extract: {e}")
                        break
            
        except Exception as e:
            logger.error(f"Discover scraping failed: {e}")
        finally:
            await browser.close()
    
    # Sort by view count (highest first) and return top results
    videos.sort(key=lambda x: x.get('view_count', 0), reverse=True)
    logger.info(f"Returning {min(len(videos), max_videos)} trending videos")
    return videos[:max_videos]


# Test function
async def test_scraper():
    """Test the scraper"""
    print("Testing tag scraper...")
    videos = await scrape_tiktok_tag('trending', max_videos=5)
    print(f"\nFound {len(videos)} videos from tag:")
    for v in videos:
        print(f"  - {v['title'][:50]}... by {v['channel_name']} | views: {v.get('view_count', 0)} | duration: {v.get('duration', 0)}s")
    
    print("\nTesting discover scraper...")
    videos = await scrape_tiktok_discover(max_videos=5)
    print(f"\nFound {len(videos)} videos from discover:")
    for v in videos:
        print(f"  - {v['title'][:50]}... by {v['channel_name']} | views: {v.get('view_count', 0)} | duration: {v.get('duration', 0)}s")


if __name__ == '__main__':
    asyncio.run(test_scraper())
