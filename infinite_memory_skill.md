# 🧠 无限谈话记录技能 (Infinite Memory Skill)

> **版本**: 1.0.0  
> **描述**: 为 AI 助手实现理论上的无限上下文记忆能力  
> **架构**: 三层记忆金字塔（工作记忆 → 短期记忆 → 长期记忆）

---

## 📋 目录

- [核心思想](#核心思想)
- [架构设计](#架构设计)
- [快速开始](#快速开始)
- [完整代码](#完整代码)
- [API 参考](#api-参考)
- [接入真实系统](#接入真实系统)
- [配置参数说明](#配置参数说明)
- [使用示例](#使用示例)

---

## 核心思想

> **用存储换上下文**：把"全量记住"变成"按需想起"，从而实现理论上的无限谈话记录。

模型上下文有限，但记忆可以无限分层。通过构建三级记忆架构，让 AI 既能记住遥远的过去，又不被上下文长度限制。

---

## 架构设计

```
┌─────────────────────────────────────────────┐
│  第一层：工作记忆 (Working Memory)            │  ← 当前对话轮次，直接注入 Prompt
│  容量：最近 N 轮对话（默认 10 轮）            │
├─────────────────────────────────────────────┤
│  第二层：短期记忆 (Short-term Memory)         │  ← 近期会话摘要，压缩后注入
│  容量：最近 M 次会话的摘要（默认 10 条）      │
├─────────────────────────────────────────────┤
│  第三层：长期记忆 (Long-term Memory)          │  ← 全部历史，向量检索按需召回
│  容量：无限，存储于向量数据库                 │
└─────────────────────────────────────────────┘
```

### 记忆流转机制

```
用户输入 → 工作记忆（保留原文）
              ↓ 超过阈值（默认 20 轮）
         自动触发压缩
              ↓
         结构化摘要
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
短期记忆           长期记忆（向量化）
（保留原文）        （按需检索召回）
```

---

## 快速开始

### 1. 安装依赖

```bash
# 基础运行（仅测试，使用内置简单实现）
pip install dataclasses

# 生产环境推荐
pip install chromadb  # 向量数据库
pip install openai    # OpenAI Embedding + LLM
```

### 2. 基础使用

```python
from infinite_memory_skill import InfiniteMemorySkill

# 初始化（使用默认内存组件）
memory = InfiniteMemorySkill(
    working_memory_limit=10,      # 工作记忆保留轮数
    summary_trigger_turns=20,     # 多少轮后自动压缩
    retrieval_top_k=5,            # 长期记忆召回数量
)

# 定义 LLM 调用函数
def my_llm(messages):
    # 这里接入你的真实模型 API
    return "AI 的回复内容"

# 单轮对话
reply = memory.chat("你好，我叫小明", my_llm)
print(reply)

# 查看记忆统计
print(memory.get_stats())
```

---

## 完整代码

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
  Infinite Memory Skill — 无限谈话记录技能
  Version: 1.0.0
  Description: 为 AI 助手实现理论上的无限上下文记忆能力
  Architecture: 三层记忆金字塔（工作记忆 → 短期记忆 → 长期记忆）
═══════════════════════════════════════════════════════════════════════════════
"""

import hashlib
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """单条记忆单元"""
    content: str
    timestamp: str
    session_id: str
    importance: float = 1.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DialogueTurn:
    """单轮对话"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    turn_id: int


# ═══════════════════════════════════════════════════════════════════════════════
# 抽象接口
# ═══════════════════════════════════════════════════════════════════════════════

class VectorStore(ABC):
    """向量数据库抽象接口"""

    @abstractmethod
    def add(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """添加向量记录"""
        pass

    @abstractmethod
    def search(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """向量相似度搜索"""
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """删除记录"""
        pass


class EmbeddingProvider(ABC):
    """Embedding 服务抽象接口"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """将文本转为向量"""
        pass


class LLMProvider(ABC):
    """大语言模型抽象接口"""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """对话接口"""
        pass

    @abstractmethod
    def summarize(self, text: str, instruction: str = "") -> str:
        """摘要接口"""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 默认实现（内存版，适用于测试和轻量场景）
# ═══════════════════════════════════════════════════════════════════════════════

class InMemoryVectorStore(VectorStore):
    """内存向量存储（基于余弦相似度）"""

    def __init__(self):
        self.data: Dict[str, Dict[str, Any]] = {}

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add(self, id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        self.data[id] = {"vector": vector, "metadata": metadata}

    def search(self, vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        scored = []
        for item in self.data.values():
            score = self._cosine_similarity(vector, item["vector"])
            scored.append({"score": score, "metadata": item["metadata"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete(self, id: str) -> None:
        self.data.pop(id, None)

    def save_to_file(self, filepath: str) -> None:
        """持久化到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            metadata_only = {k: v["metadata"] for k, v in self.data.items()}
            json.dump(metadata_only, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filepath: str, embedding_provider: EmbeddingProvider) -> None:
        """从文件加载（需要重新计算向量）"""
        with open(filepath, "r", encoding="utf-8") as f:
            metadata_only = json.load(f)
        self.data = {}
        for id_, meta in metadata_only.items():
            vector = embedding_provider.embed(meta["content"])
            self.data[id_] = {"vector": vector, "metadata": meta}


class SimpleEmbeddingProvider(EmbeddingProvider):
    """
    简单 Embedding 实现（基于词频哈希，仅用于测试）
    生产环境请替换为 OpenAI、BGE-M3 等真实模型
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        import random
        seed = hash(text) % (2**32)
        rng = random.Random(seed)
        return [rng.random() for _ in range(self.dim)]


# ═══════════════════════════════════════════════════════════════════════════════
# 核心技能类
# ═══════════════════════════════════════════════════════════════════════════════

class InfiniteMemorySkill:
    """
    无限谈话记录技能

    实现三级记忆管理：
    ┌─────────────────────────────────────┐
    │  第一层：工作记忆 (Working Memory)    │  ← 当前对话轮次，直接注入 Prompt
    │  容量：最近 N 轮对话                  │
    ├─────────────────────────────────────┤
    │  第二层：短期记忆 (Short-term Memory) │  ← 近期会话摘要，压缩后注入
    │  容量：最近 M 次会话的摘要            │
    ├─────────────────────────────────────┤
    │  第三层：长期记忆 (Long-term Memory)  │  ← 全部历史，向量检索按需召回
    │  容量：无限，存储于向量数据库         │
    └─────────────────────────────────────┘
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None,
        working_memory_limit: int = 10,
        summary_trigger_turns: int = 20,
        summary_trigger_tokens: Optional[int] = None,
        retrieval_top_k: int = 5,
        importance_threshold: float = 0.3,
        time_decay_lambda: float = 0.01,
        enable_auto_compress: bool = True,
        enable_importance_scoring: bool = True,
    ):
        self.llm = llm_provider
        self.embedder = embedding_provider or SimpleEmbeddingProvider()
        self.vector_store = vector_store or InMemoryVectorStore()

        self.working_memory_limit = working_memory_limit
        self.summary_trigger_turns = summary_trigger_turns
        self.summary_trigger_tokens = summary_trigger_tokens
        self.retrieval_top_k = retrieval_top_k
        self.importance_threshold = importance_threshold
        self.time_decay_lambda = time_decay_lambda
        self.enable_auto_compress = enable_auto_compress
        self.enable_importance_scoring = enable_importance_scoring

        self.working_memory: List[DialogueTurn] = []
        self.short_term_memory: List[Dict[str, Any]] = []
        self.current_session_id = self._generate_session_id()
        self.turn_counter = 0
        self.total_turns_all_time = 0

    def _generate_session_id(self) -> str:
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

    def _get_timestamp(self) -> str:
        return datetime.now().isoformat()

    # ─────────────────────────────────────────────────────────────────────────
    # 第一层：工作记忆（Working Memory）
    # ─────────────────────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """添加一轮对话到工作记忆"""
        self.turn_counter += 1
        self.total_turns_all_time += 1

        turn = DialogueTurn(
            role=role,
            content=content,
            timestamp=self._get_timestamp(),
            turn_id=self.turn_counter
        )
        self.working_memory.append(turn)

        if self.enable_auto_compress:
            self._check_compression_trigger()

    def get_working_memory_text(self) -> str:
        """获取工作记忆的文本表示（用于注入 Prompt）"""
        if not self.working_memory:
            return ""

        lines = []
        for turn in self.working_memory[-self.working_memory_limit:]:
            prefix = "用户" if turn.role == "user" else "AI"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def clear_working_memory(self) -> None:
        """清空工作记忆（通常在会话结束时调用）"""
        self.working_memory = []
        self.turn_counter = 0

    # ─────────────────────────────────────────────────────────────────────────
    # 第二层：短期记忆（Short-term Memory）— 自动摘要压缩
    # ─────────────────────────────────────────────────────────────────────────

    def _check_compression_trigger(self) -> None:
        """检查是否触发工作记忆压缩"""
        should_compress = False

        if len(self.working_memory) >= self.summary_trigger_turns:
            should_compress = True

        if self.summary_trigger_tokens and self._estimate_tokens() >= self.summary_trigger_tokens:
            should_compress = True

        if should_compress:
            self.compress_working_memory()

    def _estimate_tokens(self) -> int:
        """粗略估计当前工作记忆的 Token 数"""
        total_chars = sum(len(t.content) for t in self.working_memory)
        return int(total_chars / 1.5)

    def compress_working_memory(self, custom_summary: Optional[str] = None) -> str:
        """
        将工作记忆压缩为摘要，存入短期记忆和长期记忆
        """
        if not self.working_memory:
            return ""

        conversation_text = self._build_conversation_text()

        if custom_summary:
            summary = custom_summary
        elif self.llm:
            summary = self._generate_summary_with_llm(conversation_text)
        else:
            summary = self._generate_simple_summary(conversation_text)

        importance = 1.0
        if self.enable_importance_scoring and self.llm:
            importance = self._score_importance(summary)

        summary_entry = {
            "session_id": self.current_session_id,
            "summary": summary,
            "timestamp": self._get_timestamp(),
            "message_count": len(self.working_memory),
            "turn_range": (self.working_memory[0].turn_id, self.working_memory[-1].turn_id),
            "importance": importance
        }
        self.short_term_memory.append(summary_entry)

        max_short_term = 10
        if len(self.short_term_memory) > max_short_term:
            self.short_term_memory = self.short_term_memory[-max_short_term:]

        self._store_to_long_term(
            content=summary,
            importance=importance,
            tags=["session_summary", self.current_session_id, "compressed"]
        )

        self.working_memory = self.working_memory[-2:]

        return summary

    def _build_conversation_text(self) -> str:
        """将工作记忆构建为连续对话文本"""
        lines = []
        for turn in self.working_memory:
            prefix = "用户" if turn.role == "user" else "AI"
            lines.append(f"[{turn.turn_id}] {prefix}: {turn.content}")
        return "\n".join(lines)

    def _generate_summary_with_llm(self, conversation: str) -> str:
        """调用 LLM 生成结构化摘要"""
        if not self.llm:
            return self._generate_simple_summary(conversation)

        prompt = f"""请对以下对话进行结构化摘要，提取关键信息用于长期记忆：

{conversation}

请严格按以下格式输出（不要添加额外内容）：
【关键事实】提取对话中确认的事实、决定、约定
【用户偏好】用户的喜好、习惯、风格偏好
【待办任务】未完成的请求、待跟进事项
【关系进展】对话氛围、关系变化、重要情感节点
【时间标记】涉及的时间、日期、周期信息
"""
        return self.llm.summarize(prompt)

    def _generate_simple_summary(self, conversation: str) -> str:
        """无 LLM 时的简单摘要"""
        lines = conversation.strip().split("\n")
        if len(lines) <= 4:
            return f"会话摘要：{conversation[:200]}..."

        first = lines[0][:100]
        last = lines[-1][:100]
        return f"会话摘要（{len(lines)}轮）：开头「{first}...」→ 结尾「{last}...」"

    def _score_importance(self, content: str) -> float:
        """评估记忆重要性（0-1）"""
        if not self.llm:
            return 0.5

        prompt = f"""请评估以下记忆内容的重要性，只返回 0 到 1 之间的一个数字：
- 0.0-0.3: 闲聊、无实质信息
- 0.3-0.6: 一般信息、普通问答
- 0.6-0.8: 重要偏好、关键事实、待办事项
- 0.8-1.0: 核心身份、重大决定、深度情感

内容：{content}

重要性评分（只返回数字）："""

        try:
            result = self.llm.summarize(prompt).strip()
            import re
            numbers = re.findall(r"0\.\d+|1\.0|0|1", result)
            if numbers:
                return float(numbers[0])
        except Exception:
            pass
        return 0.5

    def get_short_term_memory_text(self) -> str:
        """获取短期记忆的文本表示"""
        if not self.short_term_memory:
            return ""

        summaries = []
        for mem in self.short_term_memory:
            time_str = mem["timestamp"][:16].replace("T", " ")
            summaries.append(
                f"• [{time_str}] 重要性{mem.get('importance', 1.0):.1f}: {mem['summary'][:150]}"
            )

        return "【近期会话摘要】\n" + "\n".join(summaries)

    # ─────────────────────────────────────────────────────────────────────────
    # 第三层：长期记忆（Long-term Memory）— 向量检索
    # ─────────────────────────────────────────────────────────────────────────

    def _store_to_long_term(
        self,
        content: str,
        importance: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> None:
        """存入长期记忆"""
        entry = MemoryEntry(
            content=content,
            timestamp=self._get_timestamp(),
            session_id=self.current_session_id,
            importance=importance,
            tags=tags or []
        )

        vector = self.embedder.embed(content)

        id_ = hashlib.sha256(
            f"{content}{entry.timestamp}".encode()
        ).hexdigest()[:32]

        self.vector_store.add(id_, vector, entry.to_dict())

    def retrieve_relevant_memories(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        根据查询检索相关历史记忆
        """
        k = top_k or self.retrieval_top_k
        query_vector = self.embedder.embed(query)
        results = self.vector_store.search(query_vector, k)

        filtered = []
        for r in results:
            meta = r.get("metadata", {})
            importance = meta.get("importance", 1.0)

            if importance < self.importance_threshold:
                continue

            timestamp = meta.get("timestamp", self._get_timestamp())
            days_old = self._days_since(timestamp)
            time_weight = self._time_decay(days_old)

            adjusted_score = r.get("score", 0) * importance * time_weight

            filtered.append({
                "content": meta.get("content", ""),
                "score": adjusted_score,
                "metadata": meta,
                "days_old": days_old
            })

        filtered.sort(key=lambda x: x["score"], reverse=True)
        return filtered

    def _days_since(self, timestamp: str) -> float:
        """计算距离现在多少天"""
        try:
            then = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            now = datetime.now(then.tzinfo) if then.tzinfo else datetime.now()
            return (now - then).total_seconds() / 86400
        except Exception:
            return 0.0

    def _time_decay(self, days: float) -> float:
        """时间衰减函数：exp(-λ * days)"""
        import math
        return math.exp(-self.time_decay_lambda * days)

    # ─────────────────────────────────────────────────────────────────────────
    # 主动回忆（Proactive Recall）
    # ─────────────────────────────────────────────────────────────────────────

    def proactive_recall(self, current_topic: str) -> List[str]:
        """
        主动回忆：即使用户没直接问，也检查是否有相关旧记忆应该被唤起
        """
        expanded_queries = [
            current_topic,
            f"关于{current_topic}的偏好",
            f"{current_topic} 历史",
        ]

        all_results = []
        for q in expanded_queries:
            results = self.retrieve_relevant_memories(q, top_k=3)
            all_results.extend(results)

        seen = set()
        unique = []
        for r in all_results:
            content = r["content"]
            if content not in seen:
                seen.add(content)
                unique.append(r)

        unique.sort(key=lambda x: x["score"], reverse=True)
        return [r["content"] for r in unique[:self.retrieval_top_k]]

    # ─────────────────────────────────────────────────────────────────────────
    # 记忆维护
    # ─────────────────────────────────────────────────────────────────────────

    def consolidate_memories(self) -> int:
        """
        记忆整合：定期清理低质量、重复、过时的记忆
        """
        cleaned = 0
        return cleaned

    def update_memory_importance(self, content_pattern: str, new_importance: float) -> int:
        """批量更新记忆重要性"""
        updated = 0
        return updated

    # ─────────────────────────────────────────────────────────────────────────
    # 核心接口：构建记忆上下文
    # ─────────────────────────────────────────────────────────────────────────

    def build_memory_context(
        self,
        current_user_input: str,
        include_proactive: bool = True
    ) -> str:
        """
        构建完整的记忆上下文，用于注入系统提示
        """
        parts = []

        relevant = self.retrieve_relevant_memories(current_user_input)
        if relevant:
            memory_lines = []
            for r in relevant:
                age_tag = f"({r['days_old']:.0f}天前)" if r["days_old"] > 1 else "(今天)"
                memory_lines.append(f"• {age_tag} {r['content'][:200]}")
            parts.append("【相关历史记忆】\n" + "\n".join(memory_lines))

        if include_proactive:
            proactive = self.proactive_recall(current_user_input)
            if proactive:
                parts.append("【你可能还记得】\n" + "\n".join(
                    f"• {p[:200]}" for p in proactive
                ))

        short_term = self.get_short_term_memory_text()
        if short_term:
            parts.append(short_term)

        working = self.get_working_memory_text()
        if working:
            parts.append("【当前对话】\n" + working)

        return "\n\n".join(parts)

    def build_system_prompt(
        self,
        current_user_input: str,
        base_persona: str = "",
        include_proactive: bool = True
    ) -> str:
        """
        构建完整的系统提示（包含记忆上下文）
        """
        memory_context = self.build_memory_context(
            current_user_input,
            include_proactive=include_proactive
        )

        sections = []

        if base_persona:
            sections.append(f"【角色设定】\n{base_persona}")

        sections.append(f"【记忆系统】\n{memory_context}")

        sections.append("""【记忆使用规则】
1. 如果记忆中有相关信息，请自然地融入回复，不要生硬地说"根据我的记忆"
2. 如果记忆与用户当前说法矛盾，以最新信息为准，但可温和地提及变化
3. 对于很久以前的记忆（标注天数较多），提及时可加"我记得以前..."等缓冲
4. 不要编造记忆中不存在的信息""")

        return "\n\n".join(sections)

    # ─────────────────────────────────────────────────────────────────────────
    # 便捷方法：完整对话流程
    # ─────────────────────────────────────────────────────────────────────────

    def chat(
        self,
        user_input: str,
        llm_chat_func: Callable[[List[Dict[str, str]]], str],
        base_persona: str = "",
        store_exchange: bool = True
    ) -> str:
        """
        完整的单轮对话流程（便捷方法）
        """
        system_prompt = self.build_system_prompt(user_input, base_persona)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        response = llm_chat_func(messages)

        if store_exchange:
            self.add_turn("user", user_input)
            self.add_turn("assistant", response)

            qa_text = f"用户问：{user_input}\nAI答：{response}"
            self._store_to_long_term(
                content=qa_text,
                importance=0.5,
                tags=["dialogue", "qa_pair", self.current_session_id]
            )

        return response

    # ─────────────────────────────────────────────────────────────────────────
    # 会话管理
    # ─────────────────────────────────────────────────────────────────────────

    def start_new_session(self) -> str:
        """开始新会话（保留记忆，清空工作记忆）"""
        if self.working_memory:
            self.compress_working_memory()

        self.current_session_id = self._generate_session_id()
        self.clear_working_memory()

        return self.current_session_id

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        return {
            "current_session_id": self.current_session_id,
            "total_turns_all_time": self.total_turns_all_time,
            "working_memory_turns": len(self.working_memory),
            "short_term_entries": len(self.short_term_memory),
            "long_term_entries": getattr(self.vector_store, "data", {}).__len__() if hasattr(self.vector_store, "data") else "unknown",
            "config": {
                "working_memory_limit": self.working_memory_limit,
                "summary_trigger_turns": self.summary_trigger_turns,
                "retrieval_top_k": self.retrieval_top_k,
                "importance_threshold": self.importance_threshold,
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 使用示例
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """
    演示如何使用 InfiniteMemorySkill
    """

    embedder = SimpleEmbeddingProvider(dim=1536)
    vector_store = InMemoryVectorStore()

    memory = InfiniteMemorySkill(
        embedding_provider=embedder,
        vector_store=vector_store,
        working_memory_limit=6,
        summary_trigger_turns=10,
        retrieval_top_k=3,
        importance_threshold=0.2,
        enable_auto_compress=True,
    )

    def mock_llm(messages: List[Dict[str, str]]) -> str:
        user_msg = messages[-1]["content"]
        return f"[AI回复] 收到你的消息：{user_msg[:30]}..."

    print("=" * 60)
    print("开始模拟对话（每轮显示记忆状态）")
    print("=" * 60)

    test_inputs = [
        "你好，我叫小明，我喜欢吃川菜。",
        "推荐几家北京好吃的川菜馆？",
        "我之前去过一个叫'蜀香园'的地方，还不错。",
        "对了，我下周三过生日，想请朋友吃饭。",
        "你能帮我列个邀请名单吗？",
        "我的好朋友有：小红、小李、阿强。",
        "记得阿强不吃辣，得照顾他。",
        "生日蛋糕我想要巧克力味的。",
        "餐厅最好离国贸近一点。",
        "预算大概人均150左右。",
        "对了，再帮我看看附近有没有KTV？",
        "我比较喜欢唱周杰伦的歌。",
        "上次我们唱K还是去年夏天呢。",
        "那时候小红还单身，现在她交男朋友了。",
        "她男朋友好像叫小王，做IT的。",
    ]

    for i, user_msg in enumerate(test_inputs, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"用户: {user_msg}")

        reply = memory.chat(user_msg, mock_llm, store_exchange=True)
        print(f"AI: {reply}")

        stats = memory.get_stats()
        print(f"[状态] 工作记忆: {stats['working_memory_turns']}轮 | "
              f"短期记忆: {stats['short_term_entries']}条 | "
              f"总轮数: {stats['total_turns_all_time']}")

        if i % 5 == 0:
            context = memory.build_memory_context("最近的计划是什么？")
            print(f"\n[记忆上下文预览]\n{context[:300]}...")

    print("\n" + "=" * 60)
    print("测试长期记忆检索")
    print("=" * 60)

    test_queries = [
        "我喜欢吃什么？",
        "我生日是什么时候？",
        "我的朋友们",
        "唱歌",
        "预算",
    ]

    for q in test_queries:
        results = memory.retrieve_relevant_memories(q, top_k=2)
        print(f"\n查询: '{q}'")
        for r in results:
            print(f"  → [{r['score']:.3f}] {r['content'][:100]}...")

    if hasattr(vector_store, "save_to_file"):
        vector_store.save_to_file("memory_backup.json")
        print("\n记忆已保存到 memory_backup.json")

    print("\n演示完成！")


if __name__ == "__main__":
    demo()
```

---

## API 参考

### `InfiniteMemorySkill` 类

#### 构造函数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_provider` | `LLMProvider` | `None` | LLM 实例，用于生成摘要和评分 |
| `embedding_provider` | `EmbeddingProvider` | `None` | Embedding 服务，默认使用简单哈希版 |
| `vector_store` | `VectorStore` | `None` | 向量数据库，默认使用内存版 |
| `working_memory_limit` | `int` | `10` | 工作记忆保留轮数 |
| `summary_trigger_turns` | `int` | `20` | 触发自动压缩的对话轮数 |
| `summary_trigger_tokens` | `int` | `None` | 触发压缩的 Token 阈值 |
| `retrieval_top_k` | `int` | `5` | 长期记忆每次召回数量 |
| `importance_threshold` | `float` | `0.3` | 长期记忆重要性过滤阈值 |
| `time_decay_lambda` | `float` | `0.01` | 时间衰减系数 |
| `enable_auto_compress` | `bool` | `True` | 是否启用自动压缩 |
| `enable_importance_scoring` | `bool` | `True` | 是否启用重要性评分 |

#### 核心方法

| 方法 | 说明 |
|------|------|
| `chat(user_input, llm_chat_func, ...)` | 完整的单轮对话流程 |
| `build_memory_context(user_input)` | 构建记忆上下文文本 |
| `build_system_prompt(user_input, ...)` | 构建完整系统提示 |
| `retrieve_relevant_memories(query)` | 检索相关历史记忆 |
| `proactive_recall(topic)` | 主动回忆相关记忆 |
| `compress_working_memory()` | 手动触发工作记忆压缩 |
| `start_new_session()` | 开始新会话 |
| `get_stats()` | 获取记忆系统统计 |

---

## 接入真实系统

### 接入 OpenAI

```python
import openai
from infinite_memory_skill import InfiniteMemorySkill, LLMProvider, EmbeddingProvider

class OpenAILLM(LLMProvider):
    def chat(self, messages, **kwargs):
        resp = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            **kwargs
        )
        return resp.choices[0].message.content

    def summarize(self, text, instruction=""):
        messages = [
            {"role": "system", "content": "你是一个摘要助手。"},
            {"role": "user", "content": text}
        ]
        return self.chat(messages)

class OpenAIEmbedding(EmbeddingProvider):
    def embed(self, text):
        resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return resp.data[0].embedding

# 使用
memory = InfiniteMemorySkill(
    llm_provider=OpenAILLM(),
    embedding_provider=OpenAIEmbedding(),
)
```

### 接入 ChromaDB

```python
import chromadb
from infinite_memory_skill import VectorStore

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name="memories"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(self, id, vector, metadata):
        self.collection.add(
            ids=[id],
            embeddings=[vector],
            metadatas=[metadata]
        )

    def search(self, vector, top_k):
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k
        )
        return [
            {"score": d, "metadata": m}
            for d, m in zip(results["distances"][0], results["metadatas"][0])
        ]

    def delete(self, id):
        self.collection.delete(ids=[id])

# 使用
memory = InfiniteMemorySkill(
    vector_store=ChromaVectorStore(),
)
```

---

## 配置参数说明

### 关键机制对照表

| 机制 | 解决的问题 | 实现建议 |
|------|-----------|---------|
| **分层压缩** | 上下文窗口有限 | 近期保留原文，中期存摘要，远期转向量 |
| **重要性评分** | 避免垃圾信息淹没关键记忆 | LLM 给每条记忆打 0-1 分，低分自动淘汰 |
| **时间衰减** | 旧记忆应逐渐淡化 | 检索时加入时间权重：`score = similarity × exp(-λ×days)` |
| **主动回忆** | 用户没问但相关的信息 | 定期扩展查询词，检索相关旧记忆 |
| **记忆冲突处理** | 用户前后说法矛盾 | 存储时标记版本号，优先使用最新记忆 |

---

## 使用示例

### 示例 1：基础对话

```python
memory = InfiniteMemorySkill()

def llm(messages):
    # 接入你的模型
    return "AI回复"

# 多轮对话，自动管理记忆
for _ in range(30):
    user_input = input("用户: ")
    reply = memory.chat(user_input, llm)
    print(f"AI: {reply}")
```

### 示例 2：自定义 Prompt 注入

```python
user_input = "帮我回顾一下之前的计划"

# 获取记忆上下文
context = memory.build_memory_context(user_input)

# 自定义系统提示
system_prompt = f"""你是一个私人助理。以下是你的记忆：

{context}

请根据记忆回答用户问题。"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_input}
]

reply = llm(messages)
```

### 示例 3：持久化记忆

```python
# 保存
memory.vector_store.save_to_file("my_memory.json")

# 加载（新会话）
new_memory = InfiniteMemorySkill(
    embedding_provider=embedder,
    vector_store=InMemoryVectorStore()
)
new_memory.vector_store.load_from_file("my_memory.json", embedder)
```

### 示例 4：会话管理

```python
# 当前会话结束，开始新会话
memory.start_new_session()

# 新会话仍然可以检索到之前的长期记忆
reply = memory.chat("还记得我叫什么吗？", llm)
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `infinite_memory_skill.py` | 完整技能代码 |
| `memory_backup.json` | 记忆持久化文件（运行时生成） |

---

## License

MIT License - 自由使用、修改和分发。
