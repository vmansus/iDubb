"""
Bilibili Video Uploader

B站上传流程:
1. 使用cookies登录 (SESSDATA, bili_jct, buvid3)
2. 预上传获取upload_url
3. 分片上传视频
4. 提交稿件
"""
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import aiohttp
import aiofiles
from loguru import logger

from .base import BaseUploader, UploadResult, VideoMetadata


# Upload configuration
UPLOAD_TIMEOUT_PER_CHUNK = 300  # 5 minutes per chunk
UPLOAD_TOTAL_TIMEOUT = 7200  # 2 hours max total
CHUNK_RETRY_COUNT = 5  # Number of retries per chunk (increased from 3)
CHUNK_RETRY_DELAY = 10  # Seconds to wait between retries for non-chunk operations
CHUNK_RETRY_BASE_DELAY = 10  # Base seconds for exponential backoff on chunk uploads
DEFAULT_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB default chunk size


class BilibiliUploader(BaseUploader):
    """Bilibili video uploader"""

    # API endpoints
    PREUPLOAD_URL = "https://member.bilibili.com/preupload"
    UPLOAD_URL_TEMPLATE = "https://upos-sz-upcdnbda2.bilivideo.com"
    SUBMIT_URL = "https://member.bilibili.com/x/vu/web/add/v3"
    COVER_UPLOAD_URL = "https://member.bilibili.com/x/vu/web/cover/up"

    # Complete Bilibili partition list (分区列表)
    # Format: {tid: {"name": "分区名", "parent": "父分区名", "parent_tid": 父分区ID}}
    PARTITIONS = {
        # 动画 (Animation) - tid: 1
        1: {"name": "动画", "parent": None, "parent_tid": None},
        24: {"name": "MAD·AMV", "parent": "动画", "parent_tid": 1},
        25: {"name": "MMD·3D", "parent": "动画", "parent_tid": 1},
        47: {"name": "短片·手书", "parent": "动画", "parent_tid": 1},
        257: {"name": "配音", "parent": "动画", "parent_tid": 1},
        210: {"name": "手办·模玩", "parent": "动画", "parent_tid": 1},
        86: {"name": "特摄", "parent": "动画", "parent_tid": 1},
        253: {"name": "动漫杂谈", "parent": "动画", "parent_tid": 1},
        27: {"name": "综合", "parent": "动画", "parent_tid": 1},

        # 音乐 (Music) - tid: 3
        3: {"name": "音乐", "parent": None, "parent_tid": None},
        28: {"name": "原创音乐", "parent": "音乐", "parent_tid": 3},
        31: {"name": "翻唱", "parent": "音乐", "parent_tid": 3},
        30: {"name": "VOCALOID·UTAU", "parent": "音乐", "parent_tid": 3},
        59: {"name": "演奏", "parent": "音乐", "parent_tid": 3},
        193: {"name": "MV", "parent": "音乐", "parent_tid": 3},
        29: {"name": "音乐现场", "parent": "音乐", "parent_tid": 3},
        130: {"name": "音乐综合", "parent": "音乐", "parent_tid": 3},
        243: {"name": "乐评盘点", "parent": "音乐", "parent_tid": 3},
        244: {"name": "音乐教学", "parent": "音乐", "parent_tid": 3},

        # 游戏 (Gaming) - tid: 4
        4: {"name": "游戏", "parent": None, "parent_tid": None},
        17: {"name": "单机游戏", "parent": "游戏", "parent_tid": 4},
        171: {"name": "电子竞技", "parent": "游戏", "parent_tid": 4},
        172: {"name": "手机游戏", "parent": "游戏", "parent_tid": 4},
        65: {"name": "网络游戏", "parent": "游戏", "parent_tid": 4},
        173: {"name": "桌游棋牌", "parent": "游戏", "parent_tid": 4},
        121: {"name": "GMV", "parent": "游戏", "parent_tid": 4},
        136: {"name": "音游", "parent": "游戏", "parent_tid": 4},
        19: {"name": "Mugen", "parent": "游戏", "parent_tid": 4},

        # 娱乐 (Entertainment) - tid: 5
        5: {"name": "娱乐", "parent": None, "parent_tid": None},
        71: {"name": "综艺", "parent": "娱乐", "parent_tid": 5},
        241: {"name": "娱乐杂谈", "parent": "娱乐", "parent_tid": 5},
        242: {"name": "粉丝创作", "parent": "娱乐", "parent_tid": 5},
        137: {"name": "明星综合", "parent": "娱乐", "parent_tid": 5},

        # 知识 (Knowledge) - tid: 36
        36: {"name": "知识", "parent": None, "parent_tid": None},
        201: {"name": "科学科普", "parent": "知识", "parent_tid": 36},
        124: {"name": "社科·法律·心理", "parent": "知识", "parent_tid": 36},
        228: {"name": "人文历史", "parent": "知识", "parent_tid": 36},
        207: {"name": "财经商业", "parent": "知识", "parent_tid": 36},
        208: {"name": "校园学习", "parent": "知识", "parent_tid": 36},
        209: {"name": "职业职场", "parent": "知识", "parent_tid": 36},
        229: {"name": "设计·创意", "parent": "知识", "parent_tid": 36},
        122: {"name": "野生技术协会", "parent": "知识", "parent_tid": 36},

        # 科技 (Tech) - tid: 188
        188: {"name": "科技", "parent": None, "parent_tid": None},
        95: {"name": "数码", "parent": "科技", "parent_tid": 188},
        230: {"name": "软件应用", "parent": "科技", "parent_tid": 188},
        231: {"name": "计算机技术", "parent": "科技", "parent_tid": 188},
        232: {"name": "科工机械", "parent": "科技", "parent_tid": 188},
        233: {"name": "极客DIY", "parent": "科技", "parent_tid": 188},

        # 运动 (Sports) - tid: 234
        234: {"name": "运动", "parent": None, "parent_tid": None},
        235: {"name": "篮球", "parent": "运动", "parent_tid": 234},
        249: {"name": "足球", "parent": "运动", "parent_tid": 234},
        164: {"name": "健身", "parent": "运动", "parent_tid": 234},
        236: {"name": "竞技体育", "parent": "运动", "parent_tid": 234},
        237: {"name": "运动文化", "parent": "运动", "parent_tid": 234},
        238: {"name": "运动综合", "parent": "运动", "parent_tid": 234},

        # 汽车 (Car) - tid: 223
        223: {"name": "汽车", "parent": None, "parent_tid": None},
        258: {"name": "汽车知识科普", "parent": "汽车", "parent_tid": 223},
        245: {"name": "赛车", "parent": "汽车", "parent_tid": 223},
        246: {"name": "改装玩车", "parent": "汽车", "parent_tid": 223},
        247: {"name": "新能源车", "parent": "汽车", "parent_tid": 223},
        248: {"name": "房车", "parent": "汽车", "parent_tid": 223},
        240: {"name": "摩托车", "parent": "汽车", "parent_tid": 223},
        227: {"name": "购车攻略", "parent": "汽车", "parent_tid": 223},
        176: {"name": "汽车生活", "parent": "汽车", "parent_tid": 223},

        # 生活 (Life) - tid: 160
        160: {"name": "生活", "parent": None, "parent_tid": None},
        138: {"name": "搞笑", "parent": "生活", "parent_tid": 160},
        254: {"name": "亲子", "parent": "生活", "parent_tid": 160},
        250: {"name": "出行", "parent": "生活", "parent_tid": 160},
        251: {"name": "三农", "parent": "生活", "parent_tid": 160},
        239: {"name": "家居房产", "parent": "生活", "parent_tid": 160},
        161: {"name": "手工", "parent": "生活", "parent_tid": 160},
        162: {"name": "绘画", "parent": "生活", "parent_tid": 160},
        21: {"name": "日常", "parent": "生活", "parent_tid": 160},

        # 美食 (Food) - tid: 211
        211: {"name": "美食", "parent": None, "parent_tid": None},
        76: {"name": "美食制作", "parent": "美食", "parent_tid": 211},
        212: {"name": "美食侦探", "parent": "美食", "parent_tid": 211},
        213: {"name": "美食测评", "parent": "美食", "parent_tid": 211},
        214: {"name": "田园美食", "parent": "美食", "parent_tid": 211},
        215: {"name": "美食记录", "parent": "美食", "parent_tid": 211},

        # 动物圈 (Animals) - tid: 217
        217: {"name": "动物圈", "parent": None, "parent_tid": None},
        218: {"name": "喵星人", "parent": "动物圈", "parent_tid": 217},
        219: {"name": "汪星人", "parent": "动物圈", "parent_tid": 217},
        222: {"name": "小宠异宠", "parent": "动物圈", "parent_tid": 217},
        221: {"name": "野生动物", "parent": "动物圈", "parent_tid": 217},
        220: {"name": "动物二创", "parent": "动物圈", "parent_tid": 217},
        75: {"name": "动物综合", "parent": "动物圈", "parent_tid": 217},

        # 鬼畜 (Kichiku) - tid: 119
        119: {"name": "鬼畜", "parent": None, "parent_tid": None},
        22: {"name": "鬼畜调教", "parent": "鬼畜", "parent_tid": 119},
        26: {"name": "音MAD", "parent": "鬼畜", "parent_tid": 119},
        126: {"name": "人力VOCALOID", "parent": "鬼畜", "parent_tid": 119},
        216: {"name": "鬼畜剧场", "parent": "鬼畜", "parent_tid": 119},
        127: {"name": "教程演示", "parent": "鬼畜", "parent_tid": 119},

        # 时尚 (Fashion) - tid: 155
        155: {"name": "时尚", "parent": None, "parent_tid": None},
        157: {"name": "美妆护肤", "parent": "时尚", "parent_tid": 155},
        252: {"name": "仿妆cos", "parent": "时尚", "parent_tid": 155},
        158: {"name": "穿搭", "parent": "时尚", "parent_tid": 155},
        159: {"name": "时尚潮流", "parent": "时尚", "parent_tid": 155},

        # 资讯 (Information) - tid: 202
        202: {"name": "资讯", "parent": None, "parent_tid": None},
        203: {"name": "热点", "parent": "资讯", "parent_tid": 202},
        204: {"name": "环球", "parent": "资讯", "parent_tid": 202},
        205: {"name": "社会", "parent": "资讯", "parent_tid": 202},
        206: {"name": "综合", "parent": "资讯", "parent_tid": 202},

        # 舞蹈 (Dance) - tid: 129
        129: {"name": "舞蹈", "parent": None, "parent_tid": None},
        20: {"name": "宅舞", "parent": "舞蹈", "parent_tid": 129},
        198: {"name": "街舞", "parent": "舞蹈", "parent_tid": 129},
        199: {"name": "明星舞蹈", "parent": "舞蹈", "parent_tid": 129},
        200: {"name": "中国舞", "parent": "舞蹈", "parent_tid": 129},
        154: {"name": "舞蹈综合", "parent": "舞蹈", "parent_tid": 129},
        156: {"name": "舞蹈教程", "parent": "舞蹈", "parent_tid": 129},

        # 影视 (Film & TV) - tid: 181
        181: {"name": "影视", "parent": None, "parent_tid": None},
        182: {"name": "影视杂谈", "parent": "影视", "parent_tid": 181},
        183: {"name": "影视剪辑", "parent": "影视", "parent_tid": 181},
        85: {"name": "小剧场", "parent": "影视", "parent_tid": 181},
        184: {"name": "预告·资讯", "parent": "影视", "parent_tid": 181},
        256: {"name": "短片", "parent": "影视", "parent_tid": 181},

        # 纪录片 (Documentary) - tid: 177
        177: {"name": "纪录片", "parent": None, "parent_tid": None},
        37: {"name": "人文·历史", "parent": "纪录片", "parent_tid": 177},
        178: {"name": "科学·探索·自然", "parent": "纪录片", "parent_tid": 177},
        179: {"name": "军事", "parent": "纪录片", "parent_tid": 177},
        180: {"name": "社会·美食·旅行", "parent": "纪录片", "parent_tid": 177},
    }

    # Main category mapping for simple category names
    CATEGORY_NAME_TO_TID = {
        "动画": 1, "番剧": 13, "国创": 167, "音乐": 3, "舞蹈": 129,
        "游戏": 4, "知识": 36, "科技": 188, "运动": 234, "汽车": 223,
        "生活": 160, "美食": 211, "动物圈": 217, "动物": 217, "鬼畜": 119,
        "时尚": 155, "资讯": 202, "娱乐": 5, "影视": 181, "纪录片": 177,
        "电影": 23, "电视剧": 11,
        # English aliases
        "animation": 1, "music": 3, "dance": 129, "gaming": 4, "game": 4,
        "knowledge": 36, "tech": 188, "technology": 188, "sports": 234,
        "car": 223, "automobile": 223, "life": 160, "food": 211,
        "animals": 217, "pets": 217, "fashion": 155, "entertainment": 5,
        "film": 181, "movie": 23, "tv": 11, "documentary": 177,
    }

    # Keywords for AI partition matching
    PARTITION_KEYWORDS = {
        # 知识类 - 综合知识
        36: ["教程", "教学", "学习", "知识", "科普", "解说", "分析", "讲解", "tutorial", "learn", "education",
             "AI", "人工智能", "GPT", "ChatGPT", "Claude", "大模型", "LLM", "机器学习", "深度学习",
             "神经网络", "算法", "artificial intelligence", "machine learning", "deep learning",
             "方法论", "技巧", "干货", "分享", "经验"],
        201: ["科学", "物理", "化学", "生物", "数学", "science", "physics", "chemistry"],
        124: ["心理", "法律", "社会", "psychology", "law", "social",
              "成长", "认知", "思维", "mindset", "自律", "习惯", "焦虑", "情绪", "自我",
              "personal growth", "self-improvement", "mental"],
        228: ["历史", "文化", "人文", "history", "culture", "humanities"],
        207: ["财经", "商业", "投资", "理财", "经济", "finance", "business", "investment",
              "副业", "赚钱", "收入", "财务自由", "被动收入", "money", "income", "wealth"],
        208: ["学校", "大学", "考试", "学生", "school", "university", "exam", "student"],
        209: ["职业", "工作", "职场", "career", "job", "workplace",
              "效率", "时间管理", "生产力", "productivity", "提升", "升职", "跳槽", "面试",
              "简历", "职业规划", "工作效率", "远程工作", "自由职业"],

        # 科技类
        188: ["科技", "技术", "tech", "technology", "AI", "人工智能", "GPT", "大模型", "模型", "算法",
              "artificial intelligence", "machine learning", "transformer", "neural"],
        95: ["数码", "手机", "电脑", "相机", "耳机", "phone", "computer", "camera", "digital"],
        230: ["软件", "app", "应用", "software", "application", "API", "开源", "open source"],
        231: ["编程", "代码", "开发", "programming", "code", "developer", "python", "javascript",
              "typescript", "rust", "golang", "java", "框架", "framework", "github", "coding"],
        233: ["DIY", "制作", "创客", "maker", "arduino", "raspberry"],

        # 游戏类
        4: ["游戏", "game", "gaming", "play", "gamer"],
        17: ["单机", "steam", "pc游戏", "主机", "ps5", "xbox", "switch", "roblox", "minecraft", "我的世界"],
        172: ["手游", "手机游戏", "mobile game", "王者荣耀", "原神", "genshin"],
        171: ["电竞", "比赛", "esports", "lol", "dota", "csgo", "valorant"],

        # 生活类
        160: ["生活", "日常", "vlog", "life", "daily"],
        138: ["搞笑", "幽默", "funny", "comedy", "笑话"],
        21: ["日常", "记录", "daily", "routine"],
        250: ["旅行", "旅游", "出行", "travel", "trip", "journey"],

        # 美食类
        211: ["美食", "食物", "吃", "food", "eat", "cooking", "recipe"],
        76: ["做饭", "烹饪", "菜谱", "cook", "recipe", "制作"],
        213: ["测评", "评测", "试吃", "review", "taste"],

        # 娱乐类
        5: ["娱乐", "明星", "八卦", "entertainment", "celebrity"],
        71: ["综艺", "节目", "variety", "show"],

        # 影视类
        181: ["电影", "电视剧", "影视", "film", "movie", "tv", "drama"],
        182: ["影评", "解说", "分析", "review", "analysis"],
        183: ["剪辑", "混剪", "edit", "clip"],

        # 音乐类
        3: ["音乐", "歌曲", "music", "song"],
        28: ["原创", "原创音乐", "original"],
        31: ["翻唱", "cover"],
        59: ["演奏", "钢琴", "吉他", "play", "piano", "guitar"],

        # 动物类
        217: ["动物", "宠物", "animal", "pet"],
        218: ["猫", "喵", "cat"],
        219: ["狗", "汪", "dog"],
    }

    # For backwards compatibility
    CATEGORIES = CATEGORY_NAME_TO_TID

    def __init__(self):
        super().__init__()
        self.sessdata = None
        self.bili_jct = None
        self.buvid3 = None
        self._session: Optional[aiohttp.ClientSession] = None

    def get_required_credentials(self) -> List[str]:
        return ["SESSDATA", "bili_jct", "buvid3"]

    @classmethod
    def get_partition_list(cls) -> List[Dict[str, Any]]:
        """Get full partition list for frontend"""
        result = []
        # Get main categories (parent_tid is None)
        main_categories = {tid: info for tid, info in cls.PARTITIONS.items() if info["parent_tid"] is None}

        for tid, info in sorted(main_categories.items(), key=lambda x: x[0]):
            category = {
                "tid": tid,
                "name": info["name"],
                "children": []
            }
            # Get subcategories
            for sub_tid, sub_info in cls.PARTITIONS.items():
                if sub_info["parent_tid"] == tid:
                    category["children"].append({
                        "tid": sub_tid,
                        "name": sub_info["name"]
                    })
            result.append(category)
        return result

    @classmethod
    def match_partition(cls, title: str, description: str = "", tags: List[str] = None) -> int:
        """
        Use keyword matching to find the best partition for the video.
        Returns the tid of the best matching partition.

        Args:
            title: Video title
            description: Video description
            tags: Video tags

        Returns:
            Best matching partition tid (defaults to 36 - 知识)
        """
        tags = tags or []
        # Combine all text for matching
        text = f"{title} {description} {' '.join(tags)}".lower()
        
        logger.info(f"Keyword partition matching - title: {title[:50]}...")

        # Score each partition based on keyword matches
        scores = {}
        for tid, keywords in cls.PARTITION_KEYWORDS.items():
            score = 0
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in text:
                    # Title matches are worth more
                    if keyword.lower() in title.lower():
                        score += 3
                        matched_keywords.append(f"{keyword}(title:+3)")
                    # Tag matches are worth more
                    elif any(keyword.lower() in tag.lower() for tag in tags):
                        score += 2
                        matched_keywords.append(f"{keyword}(tag:+2)")
                    else:
                        score += 1
                        matched_keywords.append(f"{keyword}(desc:+1)")
            if score > 0:
                scores[tid] = score
                if score >= 3:  # Only log significant matches
                    partition_name = cls.PARTITIONS.get(tid, {}).get('name', 'Unknown')
                    logger.debug(f"  - {partition_name}(tid={tid}): score={score}, keywords={matched_keywords}")

        if scores:
            # Log top 3 candidates
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            logger.info(f"Top partition candidates: {[(cls.PARTITIONS.get(t, {}).get('name', '?'), s) for t, s in sorted_scores]}")
            
            # Return the highest scoring partition
            best_tid = max(scores, key=scores.get)
            logger.info(f"Keyword matched partition: {cls.PARTITIONS.get(best_tid, {}).get('name', 'Unknown')} (tid={best_tid}, score={scores[best_tid]})")
            return best_tid

        # Default to 知识 (Knowledge) category
        logger.info("No partition match found, defaulting to 知识 (tid=36)")
        return 36

    @classmethod
    async def match_partition_with_ai(cls, title: str, description: str = "", tags: List[str] = None, transcript: str = "") -> int:
        """
        Use AI (LLM) to intelligently match the best partition.
        Falls back to keyword matching if AI is not available.

        Args:
            title: Video title
            description: Video description
            tags: Video tags
            transcript: Video transcript (optional, for better matching)

        Returns:
            Best matching partition tid
        """
        tags = tags or []

        # Try to use AI for matching
        try:
            from settings_store import settings_store
            settings = settings_store.load()

            # Get available translation engine API key for AI
            api_key = settings.translation.get_api_key()
            engine = settings.translation.engine
            
            logger.info(f"AI partition matching - engine: {engine}, has_api_key: {bool(api_key)}")

            if not api_key or engine in ["google", "deepl"]:
                # Fall back to keyword matching
                logger.info(f"Falling back to keyword matching (engine={engine}, has_key={bool(api_key)})")
                return cls.match_partition(title, description, tags)

            # Build partition list for AI prompt
            partition_list = "\n".join([
                f"- {tid}: {info['name']}" + (f" ({info['parent']})" if info['parent'] else "")
                for tid, info in sorted(cls.PARTITIONS.items())
                if info['parent_tid'] is not None  # Only subcategories
            ])

            prompt = f"""根据以下视频信息，选择最合适的B站分区。只返回分区ID数字，不要返回其他内容。

视频标题: {title}
视频描述: {description[:500] if description else '无'}
视频标签: {', '.join(tags) if tags else '无'}
{f"视频内容摘要: {transcript[:500]}" if transcript else ""}

分区选择指南（按内容类型选择最匹配的子分区）:

【游戏区】
- PC/主机/Steam/Roblox/Minecraft等单机游戏 → 单机游戏(17)
- 手机游戏/原神/王者荣耀 → 手机游戏(172)
- 电竞比赛/职业选手 → 电子竞技(171)
- MMO/网络游戏 → 网络游戏(65)

【知识区】
- 科学知识/科普 → 科学科普(201)
- 心理/社会/法律 → 社科·法律·心理(124)
- 历史/人文 → 人文历史(228)
- 理财/投资/商业 → 财经商业(207)
- 学习/考试/校园 → 校园学习(208)
- 职场/职业发展 → 职业职场(209)

【科技区】
- 手机/电脑/数码产品评测 → 数码(95)
- 软件/App使用 → 软件应用(230)
- 编程/AI/计算机技术 → 计算机技术(231)
- DIY/创客 → 极客DIY(233)

【生活区】
- 搞笑/幽默视频 → 搞笑(138)
- 手工制作 → 手工(161)
- 绘画/画画 → 绘画(162)
- 日常vlog/生活记录 → 日常(21)

【娱乐区】
- 综艺/娱乐节目 → 综艺(71)
- 明星/粉丝相关 → 娱乐杂谈(241)

【影视区】
- 电影/电视剧解说/评论 → 影视杂谈(182)
- 影视剪辑/混剪 → 影视剪辑(183)

【动画区】
- 动漫评论/讨论 → 动漫杂谈(253)
- MAD/AMV → MAD·AMV(24)

【音乐区】
- 翻唱 → 翻唱(31)
- 乐器演奏 → 演奏(59)
- 音乐评论 → 乐评盘点(243)

【美食区】
- 做饭/烹饪教程 → 美食制作(76)
- 吃播/美食测评 → 美食测评(213)

【动物区】
- 猫咪视频 → 喵星人(218)
- 狗狗视频 → 汪星人(219)
- 其他宠物 → 小宠异宠(222)

【运动区】
- 篮球 → 篮球(235)
- 足球 → 足球(249)
- 健身/锻炼 → 健身(164)

【汽车区】
- 汽车评测/知识 → 汽车知识科普(258)
- 新能源/电动车 → 新能源车(247)

【时尚区】
- 化妆/护肤 → 美妆护肤(157)
- 穿搭/服装 → 穿搭(158)

【鬼畜区】
- 鬼畜视频/恶搞 → 鬼畜调教(22)

【资讯区】
- 新闻/时事热点 → 热点(203)

可选分区列表:
{partition_list}

请返回最合适的分区ID（只返回数字）:"""

            # Use the translation engine to get AI response
            if engine in ["gpt", "openai"]:
                import openai
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                tid_str = response.choices[0].message.content.strip()
            elif engine in ["claude", "anthropic"]:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": prompt}]
                )
                tid_str = response.content[0].text.strip()
            elif engine == "deepseek":
                import openai
                client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0.1
                )
                tid_str = response.choices[0].message.content.strip()
            else:
                return cls.match_partition(title, description, tags)

            # Parse the response
            logger.info(f"AI response for partition: '{tid_str}'")
            try:
                tid = int(tid_str)
                if tid in cls.PARTITIONS:
                    logger.info(f"AI matched partition: {cls.PARTITIONS[tid]['name']} (tid={tid})")
                    return tid
            except ValueError:
                pass

            # Fall back to keyword matching
            return cls.match_partition(title, description, tags)

        except Exception as e:
            logger.warning(f"AI partition matching failed: {e}, falling back to keyword matching")
            return cls.match_partition(title, description, tags)

    async def authenticate(self, credentials: Dict[str, str]) -> bool:
        """
        Authenticate using cookies

        Args:
            credentials: Dict with SESSDATA, bili_jct, buvid3
        """
        try:
            self.sessdata = credentials.get("SESSDATA")
            self.bili_jct = credentials.get("bili_jct")
            self.buvid3 = credentials.get("buvid3")

            if not all([self.sessdata, self.bili_jct, self.buvid3]):
                logger.error("Missing required Bilibili credentials")
                return False

            # Create session with cookies
            cookies = {
                "SESSDATA": self.sessdata,
                "bili_jct": self.bili_jct,
                "buvid3": self.buvid3,
            }

            # Create SSL context that doesn't verify certificates (for macOS compatibility)
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            # Configure connector with keep-alive and connection limits
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                limit=10,
                limit_per_host=5,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
            )

            # Configure timeout for general requests
            timeout = aiohttp.ClientTimeout(
                total=60,  # General timeout for non-upload requests
                connect=30,
                sock_read=60,
            )

            self._session = aiohttp.ClientSession(
                connector=connector,
                cookies=cookies,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Referer": "https://member.bilibili.com/",
                    "Connection": "keep-alive",
                }
            )

            # Verify authentication by checking user info
            async with self._session.get(
                "https://api.bilibili.com/x/web-interface/nav"
            ) as resp:
                data = await resp.json()
                if data.get("code") == 0 and data.get("data", {}).get("isLogin"):
                    self._authenticated = True
                    username = data["data"].get("uname", "Unknown")
                    logger.info(f"Bilibili authenticated as: {username}")
                    return True
                else:
                    logger.error(f"Bilibili auth failed: {data}")
                    return False

        except Exception as e:
            logger.error(f"Bilibili authentication error: {e}")
            return False

    async def _verify_login(self) -> bool:
        """Verify that cookies are still valid"""
        try:
            verify_timeout = aiohttp.ClientTimeout(total=30, connect=15)
            async with self._session.get(
                "https://api.bilibili.com/x/web-interface/nav",
                timeout=verify_timeout,
            ) as resp:
                data = await resp.json()
                if data.get("code") == 0 and data.get("data", {}).get("isLogin"):
                    username = data["data"].get("uname", "Unknown")
                    vip_type = data["data"].get("vipType", 0)
                    logger.info(f"Login verified: {username} (VIP type: {vip_type})")
                    return True
                else:
                    logger.error(f"Login verification failed: {data}")
                    return False
        except asyncio.TimeoutError:
            logger.error("Login verification timeout")
            return False
        except Exception as e:
            logger.error(f"Login verification error: {e}")
            return False

    async def _preupload(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """Get preupload info with retry"""
        preupload_timeout = aiohttp.ClientTimeout(total=60, connect=30)

        for attempt in range(CHUNK_RETRY_COUNT):
            try:
                file_size = video_path.stat().st_size
                filename = video_path.name

                params = {
                    "name": filename,
                    "size": file_size,
                    "r": "upos",
                    "profile": "ugcupos/bup",
                    "ssl": "0",
                    "version": "2.14.0",
                    "build": "2140000",
                    "upcdn": "bda2",
                    "probe_version": "20221109",
                }

                async with self._session.get(
                    self.PREUPLOAD_URL,
                    params=params,
                    timeout=preupload_timeout,
                ) as resp:
                    data = await resp.json()
                    logger.info(f"Preupload response: upos_uri={data.get('upos_uri')}, "
                               f"biz_id={data.get('biz_id')}, endpoint={data.get('endpoint')}, "
                               f"chunk_size={data.get('chunk_size', 'default')}")
                    if "upos_uri" in data:
                        return data
                    else:
                        logger.error(f"Preupload failed (attempt {attempt + 1}/{CHUNK_RETRY_COUNT}): {data}")
                        if attempt < CHUNK_RETRY_COUNT - 1:
                            await asyncio.sleep(CHUNK_RETRY_DELAY)
                            continue
                        return None

            except asyncio.TimeoutError:
                logger.warning(f"Preupload timeout (attempt {attempt + 1}/{CHUNK_RETRY_COUNT})")
                if attempt < CHUNK_RETRY_COUNT - 1:
                    await asyncio.sleep(CHUNK_RETRY_DELAY)
            except Exception as e:
                logger.error(f"Preupload error (attempt {attempt + 1}/{CHUNK_RETRY_COUNT}): {e}")
                if attempt < CHUNK_RETRY_COUNT - 1:
                    await asyncio.sleep(CHUNK_RETRY_DELAY)

        logger.error("Preupload failed after all retries")
        return None

    async def _upload_chunk_with_retry(
        self,
        upload_url: str,
        chunk_data: bytes,
        chunk_num: int,
        total_chunks: int,
        upload_id: str,
        file_size: int,
        auth: str,
    ) -> Optional[str]:
        """Upload a single chunk with retry logic"""
        start = chunk_num * len(chunk_data)
        end = min(start + len(chunk_data), file_size)

        # Timeout for chunk upload (scales with chunk size)
        chunk_timeout = aiohttp.ClientTimeout(
            total=UPLOAD_TIMEOUT_PER_CHUNK,
            connect=30,
            sock_read=UPLOAD_TIMEOUT_PER_CHUNK,
        )

        for attempt in range(CHUNK_RETRY_COUNT):
            try:
                start_time = time.time()
                async with self._session.put(
                    upload_url,
                    params={
                        "partNumber": chunk_num + 1,
                        "uploadId": upload_id,
                        "chunk": chunk_num,
                        "chunks": total_chunks,
                        "size": len(chunk_data),
                        "start": start,
                        "end": end,
                        "total": file_size,
                    },
                    data=chunk_data,
                    headers={"X-Upos-Auth": auth},
                    timeout=chunk_timeout,
                ) as resp:
                    elapsed = time.time() - start_time
                    if resp.status == 200:
                        etag = resp.headers.get("ETag", f"etag{chunk_num + 1}")
                        speed = len(chunk_data) / elapsed / 1024 / 1024 if elapsed > 0 else 0
                        logger.info(f"Chunk {chunk_num + 1}/{total_chunks} uploaded successfully "
                                   f"({len(chunk_data)/1024/1024:.1f}MB in {elapsed:.1f}s, {speed:.2f}MB/s)")
                        return etag
                    else:
                        resp_text = await resp.text()
                        logger.warning(f"Chunk {chunk_num + 1} upload failed (attempt {attempt + 1}/{CHUNK_RETRY_COUNT}): "
                                      f"status={resp.status}, response={resp_text[:200]}")

            except asyncio.TimeoutError:
                logger.warning(f"Chunk {chunk_num + 1} upload timeout (attempt {attempt + 1}/{CHUNK_RETRY_COUNT})")
            except aiohttp.ClientError as e:
                logger.warning(f"Chunk {chunk_num + 1} upload network error (attempt {attempt + 1}/{CHUNK_RETRY_COUNT}): {e}")
            except Exception as e:
                logger.warning(f"Chunk {chunk_num + 1} upload error (attempt {attempt + 1}/{CHUNK_RETRY_COUNT}): {e}")

            if attempt < CHUNK_RETRY_COUNT - 1:
                # Exponential backoff: 10s, 20s, 40s, 80s...
                delay = CHUNK_RETRY_BASE_DELAY * (2 ** attempt)
                logger.info(f"Retrying chunk {chunk_num + 1} in {delay} seconds (attempt {attempt + 2}/{CHUNK_RETRY_COUNT})...")
                await asyncio.sleep(delay)

        logger.error(f"Chunk {chunk_num + 1} upload failed after {CHUNK_RETRY_COUNT} attempts")
        return None

    async def _upload_chunks(
        self,
        video_path: Path,
        preupload_info: Dict[str, Any],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Optional[str]:
        """Upload video in chunks with retry support"""
        try:
            upos_uri = preupload_info["upos_uri"]
            auth = preupload_info["auth"]
            biz_id = preupload_info["biz_id"]
            # Use larger chunk size for better performance (10MB default, or server-specified)
            chunk_size = preupload_info.get("chunk_size", DEFAULT_CHUNK_SIZE)

            # Parse upload URL
            upload_url = f"https:{preupload_info['endpoint']}/{upos_uri.replace('upos://', '')}"

            file_size = video_path.stat().st_size
            total_chunks = (file_size + chunk_size - 1) // chunk_size

            logger.info(f"Starting upload: file_size={file_size/1024/1024:.1f}MB, "
                       f"chunk_size={chunk_size/1024/1024:.1f}MB, total_chunks={total_chunks}")

            # Initialize upload with timeout
            init_timeout = aiohttp.ClientTimeout(total=60, connect=30)
            try:
                async with self._session.post(
                    upload_url,
                    params={
                        "uploads": "",
                        "output": "json",
                    },
                    headers={"X-Upos-Auth": auth},
                    timeout=init_timeout,
                ) as resp:
                    init_data = await resp.json()
                    upload_id = init_data.get("upload_id")
                    logger.info(f"Upload initialized: upload_id={upload_id}")
            except asyncio.TimeoutError:
                logger.error("Upload initialization timeout")
                return None
            except Exception as e:
                logger.error(f"Upload initialization failed: {e}")
                return None

            if not upload_id:
                logger.error("Failed to get upload_id from initialization response")
                return None

            # Upload chunks and collect ETags
            parts = []
            upload_start_time = time.time()

            async with aiofiles.open(video_path, "rb") as f:
                for chunk_num in range(total_chunks):
                    chunk_data = await f.read(chunk_size)

                    etag = await self._upload_chunk_with_retry(
                        upload_url=upload_url,
                        chunk_data=chunk_data,
                        chunk_num=chunk_num,
                        total_chunks=total_chunks,
                        upload_id=upload_id,
                        file_size=file_size,
                        auth=auth,
                    )

                    if etag is None:
                        return None

                    parts.append({"partNumber": chunk_num + 1, "eTag": etag})

                    # Report progress
                    progress = (chunk_num + 1) / total_chunks * 100
                    elapsed = time.time() - upload_start_time
                    uploaded_bytes = (chunk_num + 1) * chunk_size
                    if uploaded_bytes > file_size:
                        uploaded_bytes = file_size
                    avg_speed = uploaded_bytes / elapsed / 1024 / 1024 if elapsed > 0 else 0

                    logger.info(f"Upload progress: {progress:.1f}% ({chunk_num + 1}/{total_chunks}), "
                               f"avg speed: {avg_speed:.2f}MB/s")

                    if progress_callback:
                        progress_callback(chunk_num + 1, total_chunks, progress)

            # Complete upload - include collected parts with retry
            total_elapsed = time.time() - upload_start_time
            avg_speed = file_size / total_elapsed / 1024 / 1024 if total_elapsed > 0 else 0
            logger.info(f"All chunks uploaded. Total time: {total_elapsed:.1f}s, avg speed: {avg_speed:.2f}MB/s")
            logger.info(f"Completing upload with {len(parts)} parts")

            complete_timeout = aiohttp.ClientTimeout(total=120, connect=30)
            for attempt in range(CHUNK_RETRY_COUNT):
                try:
                    async with self._session.post(
                        upload_url,
                        params={
                            "output": "json",
                            "name": video_path.name,
                            "profile": "ugcupos/bup",
                            "uploadId": upload_id,
                            "biz_id": biz_id,
                        },
                        json={"parts": parts},
                        headers={"X-Upos-Auth": auth},
                        timeout=complete_timeout,
                    ) as resp:
                        complete_data = await resp.json()
                        logger.info(f"Upload complete response: {complete_data}")
                        if complete_data.get("OK") == 1:
                            # Return the filename without extension
                            filename = upos_uri.replace("upos://", "").split("/")[-1].rsplit(".", 1)[0]
                            logger.info(f"Upload complete, filename: {filename}, biz_id: {biz_id}")
                            return filename
                        else:
                            logger.error(f"Upload complete failed: {complete_data}")
                            if attempt < CHUNK_RETRY_COUNT - 1:
                                await asyncio.sleep(CHUNK_RETRY_DELAY)
                                continue
                            return None
                except asyncio.TimeoutError:
                    logger.warning(f"Upload completion timeout (attempt {attempt + 1}/{CHUNK_RETRY_COUNT})")
                    if attempt < CHUNK_RETRY_COUNT - 1:
                        await asyncio.sleep(CHUNK_RETRY_DELAY)
                except Exception as e:
                    logger.warning(f"Upload completion error (attempt {attempt + 1}/{CHUNK_RETRY_COUNT}): {e}")
                    if attempt < CHUNK_RETRY_COUNT - 1:
                        await asyncio.sleep(CHUNK_RETRY_DELAY)

            logger.error("Upload completion failed after all retries")
            return None

        except Exception as e:
            logger.error(f"Upload chunks error: {e}", exc_info=True)
            return None

    async def _upload_cover(self, cover_path: Path) -> Optional[str]:
        """Upload video cover image with retry"""
        cover_timeout = aiohttp.ClientTimeout(total=60, connect=30)

        for attempt in range(CHUNK_RETRY_COUNT):
            try:
                async with aiofiles.open(cover_path, "rb") as f:
                    cover_data = await f.read()

                import base64
                cover_base64 = base64.b64encode(cover_data).decode()
                logger.info(f"Uploading cover: {cover_path.name} ({len(cover_data)/1024:.1f}KB)")

                async with self._session.post(
                    self.COVER_UPLOAD_URL,
                    data={
                        "cover": f"data:image/jpeg;base64,{cover_base64}",
                        "csrf": self.bili_jct,
                    },
                    timeout=cover_timeout,
                ) as resp:
                    data = await resp.json()
                    if data.get("code") == 0:
                        return data["data"]["url"]
                    else:
                        logger.warning(f"Cover upload failed (attempt {attempt + 1}/{CHUNK_RETRY_COUNT}): {data}")
                        if attempt < CHUNK_RETRY_COUNT - 1:
                            await asyncio.sleep(CHUNK_RETRY_DELAY)
                            continue
                        return None

            except asyncio.TimeoutError:
                logger.warning(f"Cover upload timeout (attempt {attempt + 1}/{CHUNK_RETRY_COUNT})")
                if attempt < CHUNK_RETRY_COUNT - 1:
                    await asyncio.sleep(CHUNK_RETRY_DELAY)
            except Exception as e:
                logger.error(f"Cover upload error (attempt {attempt + 1}/{CHUNK_RETRY_COUNT}): {e}")
                if attempt < CHUNK_RETRY_COUNT - 1:
                    await asyncio.sleep(CHUNK_RETRY_DELAY)

        logger.error("Cover upload failed after all retries")
        return None

    async def upload(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        source_url: str = ""
    ) -> UploadResult:
        """Upload video to Bilibili"""
        if not self._authenticated:
            return UploadResult(
                success=False,
                platform="bilibili",
                error="Not authenticated"
            )

        try:
            # Load Bilibili settings (use reload to ensure latest settings)
            from settings_store import settings_store
            settings = settings_store.reload()  # Force reload from disk to get latest settings
            bilibili_settings = settings.bilibili
            logger.info(f"Loaded Bilibili settings: default_tid={bilibili_settings.default_tid}, "
                       f"auto_match_partition={bilibili_settings.auto_match_partition}, "
                       f"is_original={bilibili_settings.is_original}")

            # Step 0: Verify login is still valid
            logger.info("Verifying Bilibili login...")
            if not await self._verify_login():
                return UploadResult(
                    success=False,
                    platform="bilibili",
                    error="Cookie expired or invalid. Please re-login to Bilibili."
                )

            # Log video file info
            file_size = video_path.stat().st_size
            logger.info(f"Starting Bilibili upload: file={video_path.name}, size={file_size/1024/1024:.1f}MB")

            # Step 1: Preupload
            logger.info("Step 1: Getting preupload info...")
            preupload_info = await self._preupload(video_path)
            if not preupload_info:
                return UploadResult(
                    success=False,
                    platform="bilibili",
                    error="Preupload failed - unable to get upload URL from Bilibili"
                )

            # Step 2: Upload video chunks
            logger.info("Step 2: Uploading video chunks...")
            filename = await self._upload_chunks(video_path, preupload_info)
            if not filename:
                return UploadResult(
                    success=False,
                    platform="bilibili",
                    error="Video upload failed - chunk upload or completion error"
                )

            # Step 3: Upload cover if provided
            logger.info("Step 3: Uploading cover image...")
            cover_url = ""
            if metadata.cover_path and metadata.cover_path.exists():
                cover_url = await self._upload_cover(metadata.cover_path) or ""
                if cover_url:
                    logger.info(f"Cover uploaded: {cover_url[:60]}...")
                else:
                    logger.warning("Cover upload failed, continuing without cover")

            # Step 4: Determine partition (tid)
            logger.info("Step 4: Determining video partition...")
            if bilibili_settings.auto_match_partition:
                # Use AI/keyword matching first when enabled
                tid = await self.match_partition_with_ai(
                    title=metadata.title,
                    description=metadata.description,
                    tags=metadata.tags
                )
                logger.info(f"AI matched partition: {self.PARTITIONS.get(tid, {}).get('name', 'Unknown')} (tid={tid})")
            elif bilibili_settings.default_tid > 0:
                # Fall back to configured default partition
                tid = bilibili_settings.default_tid
                logger.info(f"Using configured partition: {self.PARTITIONS.get(tid, {}).get('name', 'Unknown')} (tid={tid})")
            else:
                # Default to 知识 category
                tid = 36
                logger.info(f"Using default partition: 知识 (tid=36)")

            # Step 5: Determine copyright (自制 vs 转载)
            is_original = bilibili_settings.is_original

            # For 转载 content, source URL is required
            source = ""
            if not is_original:
                source = source_url or metadata.source_url or "转载"

            submit_data = {
                "copyright": 1 if is_original else 2,
                "videos": [{
                    "filename": filename,
                    "title": metadata.title[:80],  # Max 80 chars
                    "desc": "",
                    "cid": preupload_info["biz_id"],
                }],
                "source": source,
                "tid": tid,
                "cover": cover_url,
                "title": metadata.title[:80],
                "tag": ",".join(metadata.tags[:12]),  # Max 12 tags
                "desc_format_id": 0,
                "desc": metadata.description[:2000],  # Max 2000 chars
                "dynamic": "",
                "interactive": 0,
                "no_reprint": 1 if is_original else 0,
                "open_elec": 0,
                "csrf": self.bili_jct,
            }

            logger.info("Step 5: Submitting video to Bilibili...")
            logger.info(f"Submit data: filename={submit_data['videos'][0]['filename']}, "
                       f"cid={submit_data['videos'][0]['cid']}, tid={tid}, "
                       f"cover={'yes' if cover_url else 'no'}, copyright={'自制' if is_original else '转载'}")

            submit_timeout = aiohttp.ClientTimeout(total=120, connect=30)
            async with self._session.post(
                self.SUBMIT_URL,
                json=submit_data,
                params={"csrf": self.bili_jct},
                timeout=submit_timeout,
            ) as resp:
                result = await resp.json()
                logger.info(f"Submit response: code={result.get('code')}, message={result.get('message')}")

                if result.get("code") == 0:
                    bvid = result["data"]["bvid"]
                    aid = result["data"]["aid"]
                    logger.info(f"Bilibili upload success! BVID: {bvid}")
                    return UploadResult(
                        success=True,
                        video_id=bvid,
                        video_url=f"https://www.bilibili.com/video/{bvid}",
                        platform="bilibili",
                        extra_info={"aid": aid, "bvid": bvid}
                    )
                else:
                    logger.error(f"Submit failed: {result}")
                    return UploadResult(
                        success=False,
                        platform="bilibili",
                        error=result.get("message", "Submit failed")
                    )

        except Exception as e:
            logger.error(f"Bilibili upload error: {e}")
            return UploadResult(
                success=False,
                platform="bilibili",
                error=str(e)
            )

    async def check_upload_status(self, video_id: str) -> Dict[str, Any]:
        """Check video processing status"""
        try:
            async with self._session.get(
                "https://member.bilibili.com/x/web/archive/view",
                params={"bvid": video_id}
            ) as resp:
                data = await resp.json()
                if data.get("code") == 0:
                    state = data["data"]["archive"]["state"]
                    # state: 0=审核中, 1=已通过, -1=未通过, -4=已删除
                    return {
                        "status": "processing" if state == 0 else "published" if state == 1 else "failed",
                        "state": state,
                        "data": data["data"]
                    }
                return {"status": "unknown", "error": data.get("message")}

        except Exception as e:
            logger.error(f"Status check error: {e}")
            return {"status": "error", "error": str(e)}

    async def close(self):
        """Close session"""
        if self._session:
            await self._session.close()
