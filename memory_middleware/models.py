"""数据模型：

- UserProfile: 长期记忆（用户画像），增量总结的结构化目标
- MemoryFragmentsMetadata / MemoryFragments: 暂存库片段（含主题/类型/时间/巩固次数）
- SummaryMemoryAi / SummaryMemoryFragmentsConfig: 切分模型的结构化输出
- TimeMemoryFormulaParam: 数学调参模型的结构化输出
"""
from datetime import datetime
from typing import Literal, Optional, List, Union

from pydantic import BaseModel, Field

# identity: 包含用户身份信息（姓名、职业等）
# preference: 包含用户偏好、习惯
# decision: 包含关键决策、承诺
# fact: 包含客观事实、技术细节
# episode: 描述某个经历或任务过程
# chat: 一般性闲聊，长期价值低
fragments_type = Literal["identity", "preference", "decision", "fact", "episode", "chat"]


class UserProfile(BaseModel):
    """结构化用户画像，用于大模型跨会话记忆推理"""
    # --- 核心身份 ---
    user_name: str = Field(default='', description="用户名")
    user_sex: Literal['男', '女', '保密'] = Field(default='保密', description="用户性别")
    user_age: Optional[int] = Field(default=None, description="用户年龄")
    user_occupation: Optional[str] = Field(default=None, description="职业")

    # --- 偏好与风格 ---
    user_hobby: list[str] = Field(default=[], description="用户爱好列表")
    user_interest: list[str] = Field(default=[], description="用户兴趣列表")
    user_speak_style: str = Field(default='', description="用户说话风格")
    user_communication_style: Optional[
        Literal['直截了当', '委婉', '幽默', '正式']
    ] | str = Field(default='', description="用户沟通风格")
    # --- 知识与能力 ---
    user_expertise: list[str] = Field(default_factory=list, description="专业领域")

    # --- 当前上下文 ---
    user_goals: list[str] = Field(default_factory=list, description="当前目标/正在进行的项目")
    user_constraints: list[str] = Field(default_factory=list, description="明确约束条件")
    user_values: list[str] = Field(default_factory=list, description="核心价值观")

    # --- 关系网络 ---
    key_relationships: dict[str, str] = Field(default_factory=dict, description="重要关系人与角色")

    # --- 业务画像 ---
    user_person: Optional[dict] = Field(
        default=None,
        description="业务画像（可选，业务定制字段，如偏好/厌恶/喜欢的商品）")

    # --- 元数据 ---
    last_updated: Optional[datetime] = Field(
        default=None, description="最后更新时间，系统会自动补充，无需添加")


class MemoryFragmentsMetadata(BaseModel):
    theme: str = Field(default='', description="记忆主题")
    type: List[fragments_type] = Field(default=[], description="在同一主题下可能存在的类型")
    time: float = Field(default=0.0, description="记忆时间（unix 秒）")
    strengthen_num: int = Field(default=0, description="巩固次数")


class MemoryFragments(BaseModel):
    """暂存库片段（切分模型的产出，用于向量 RAG）"""
    config: MemoryFragmentsMetadata = Field(default_factory=MemoryFragmentsMetadata)
    content: str = Field(default='', description="记忆内容")


class SummaryMemoryFragmentsConfig(BaseModel):
    theme: str = Field(default='', description="记忆主题")
    type: List[fragments_type] = Field(default=[], description="在同一主题下可能存在的类型")
    scope: str = Field(
        default='',
        description="记忆范围（从0开始计数）要严格按照格式输入。格式：start-end，如：0-50；51-100")


class SummaryMemoryAi(BaseModel):
    theme_num: int = Field(default=0, description="长对话的主题总数，必须与config列表长度一致")
    config: List[SummaryMemoryFragmentsConfig] = Field(
        default_factory=list,
        description="记忆片段配置列表，每个配置对应一个主题的摘要规则")
    current_theme: str = Field(default='', description="当前正在处理的主题")


class TimeMemoryFormulaParam(BaseModel):
    """数学调参模型的结构化输出（α+β+γ=1，w0+w1+w2=1）"""
    w0: float = Field(default=0.1, description='基础重要性')
    w1: float = Field(default=0.8, description='类型重要性的权重')
    w2: float = Field(default=0.1, description='巩固次数的权重')
    alpha: float = Field(default=0.33, description='语义相关性权重')
    beta: float = Field(default=0.33, description='时间衰减权重')
    gamma: float = Field(default=0.34, description='固有重要性权重')
    delta: float = Field(default=0.0, description='基线保留偏移，仅用于微调')


# Union 仅为类型引用便利（某些推理后端需要完整的 Literal 类型）
MemoryType = Union[
    Literal['identity'], Literal['preference'], Literal['decision'],
    Literal['fact'], Literal['episode'], Literal['chat'],
]
