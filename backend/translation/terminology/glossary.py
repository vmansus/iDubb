"""
Glossary Data Structures

Defines glossary entries and collections for terminology management.
"""
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from loguru import logger


@dataclass
class GlossaryEntry:
    """A single glossary entry with source term and translation"""
    source_term: str
    target_term: str
    source_lang: str = "en"
    target_lang: str = "zh-CN"
    category: Optional[str] = None  # e.g., "tech", "name", "brand"
    context: Optional[str] = None   # Usage context
    case_sensitive: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def matches(self, text: str) -> bool:
        """Check if this entry matches in the text"""
        if self.case_sensitive:
            return self.source_term in text
        return self.source_term.lower() in text.lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "source_term": self.source_term,
            "target_term": self.target_term,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "category": self.category,
            "context": self.context,
            "case_sensitive": self.case_sensitive,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlossaryEntry":
        """Create from dictionary"""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        return cls(
            source_term=data["source_term"],
            target_term=data["target_term"],
            source_lang=data.get("source_lang", "en"),
            target_lang=data.get("target_lang", "zh-CN"),
            category=data.get("category"),
            context=data.get("context"),
            case_sensitive=data.get("case_sensitive", False),
            created_at=created_at,
        )


@dataclass
class Glossary:
    """A collection of glossary entries"""
    name: str
    description: str = ""
    source_lang: str = "en"
    target_lang: str = "zh-CN"
    entries: List[GlossaryEntry] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_entry(
        self,
        source_term: str,
        target_term: str,
        category: Optional[str] = None,
        context: Optional[str] = None,
        case_sensitive: bool = False
    ) -> GlossaryEntry:
        """Add a new entry to the glossary"""
        entry = GlossaryEntry(
            source_term=source_term,
            target_term=target_term,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            category=category,
            context=context,
            case_sensitive=case_sensitive,
        )
        self.entries.append(entry)
        self.updated_at = datetime.now()
        return entry

    def remove_entry(self, source_term: str) -> bool:
        """Remove an entry by source term"""
        for i, entry in enumerate(self.entries):
            if entry.source_term.lower() == source_term.lower():
                del self.entries[i]
                self.updated_at = datetime.now()
                return True
        return False

    def get_entry(self, source_term: str) -> Optional[GlossaryEntry]:
        """Get entry by source term"""
        for entry in self.entries:
            if entry.source_term.lower() == source_term.lower():
                return entry
        return None

    def find_matches(self, text: str) -> List[GlossaryEntry]:
        """Find all entries that match in the text"""
        return [entry for entry in self.entries if entry.matches(text)]

    def to_dict_mapping(self) -> Dict[str, str]:
        """Get simple source->target mapping"""
        return {e.source_term: e.target_term for e in self.entries}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "entries": [e.to_dict() for e in self.entries],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Glossary":
        """Create from dictionary"""
        entries = [GlossaryEntry.from_dict(e) for e in data.get("entries", [])]

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        else:
            updated_at = datetime.now()

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            source_lang=data.get("source_lang", "en"),
            target_lang=data.get("target_lang", "zh-CN"),
            entries=entries,
            created_at=created_at,
            updated_at=updated_at,
        )

    def save(self, file_path: Path) -> bool:
        """Save glossary to JSON file"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved glossary '{self.name}' to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save glossary: {e}")
            return False

    @classmethod
    def load(cls, file_path: Path) -> Optional["Glossary"]:
        """Load glossary from JSON file"""
        try:
            file_path = Path(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            glossary = cls.from_dict(data)
            logger.info(f"Loaded glossary '{glossary.name}' with {len(glossary.entries)} entries")
            return glossary
        except Exception as e:
            logger.error(f"Failed to load glossary: {e}")
            return None

    def __len__(self) -> int:
        return len(self.entries)


# Built-in glossaries for common domains
TECH_GLOSSARY_EN_ZH = {
    # Programming terms
    "API": "API接口",
    "backend": "后端",
    "frontend": "前端",
    "database": "数据库",
    "server": "服务器",
    "client": "客户端",
    "algorithm": "算法",
    "function": "函数",
    "variable": "变量",
    "parameter": "参数",
    "framework": "框架",
    "library": "库",
    "deployment": "部署",
    "repository": "代码仓库",
    "commit": "提交",
    "branch": "分支",
    "merge": "合并",
    "pull request": "拉取请求",
    "code review": "代码审查",
    "debugging": "调试",
    "refactoring": "重构",

    # AI/ML terms
    "machine learning": "机器学习",
    "deep learning": "深度学习",
    "neural network": "神经网络",
    "training": "训练",
    "inference": "推理",
    "model": "模型",
    "dataset": "数据集",
    "hyperparameter": "超参数",
    "overfitting": "过拟合",
    "underfitting": "欠拟合",
    "transformer": "Transformer模型",
    "attention": "注意力机制",
    "embedding": "嵌入",
    "fine-tuning": "微调",
    "prompt": "提示词",
    "token": "令牌",

    # Video/Media terms
    "subtitle": "字幕",
    "dubbing": "配音",
    "transcription": "转录",
    "translation": "翻译",
    "TTS": "文字转语音",
    "voice cloning": "语音克隆",
    "video editing": "视频编辑",
    "rendering": "渲染",
    "compression": "压缩",
    "codec": "编解码器",
    "frame rate": "帧率",
    "resolution": "分辨率",
    "bitrate": "比特率",
}


def create_tech_glossary(source_lang: str = "en", target_lang: str = "zh-CN") -> Glossary:
    """Create a pre-built tech glossary"""
    glossary = Glossary(
        name="Tech Glossary",
        description="Common technology and programming terms",
        source_lang=source_lang,
        target_lang=target_lang,
    )

    for source, target in TECH_GLOSSARY_EN_ZH.items():
        glossary.add_entry(
            source_term=source,
            target_term=target,
            category="tech",
            case_sensitive=False
        )

    return glossary
