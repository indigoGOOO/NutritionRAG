"""记忆层 - 让Agent具备记忆能力

三种记忆类型：
- conversation: 对话历史，记录问答轮次
- profile: 用户画像，偏好/过敏/限制
- knowledge: 长期知识，有价值问答对复用
"""

from src.memory.memory_manager import MemoryManager

__all__ = ["MemoryManager"]
