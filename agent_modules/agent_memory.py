from langchain_core.memory import BaseMemory
from langchain_core.pydantic_v1 import Field
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from .agent_retriever import AgentRetriever
from datetime import datetime
from typing import Any, Dict, List, Optional
import re

class AgentMemory(BaseMemory):
    name: str # Agent별 working Memory
    memory_retriever: AgentRetriever # Agent별 Long-term Memory
    reflection_threshold: Optional[float] = 1.5 # 단기 기억의 중요도가 쌓일 때, 이를 종합하여 장기기억으로 전환하기 위한 임계치
    importance_weight: float = 0.15 # 기억의 가중치 (Default: 0.15)
    aggregate_importance: float = 0.0 # 저장된 기억의 누적 가중치
    max_tokens_limit: int = 512 # 기억을 load하는 것과 관련된 사용 토큰 제한
    reflecting: bool = False # 장기기억 전환(성찰) 여부
    working_memory: List[Document] = Field(default_factory=list) # Working Memory
    # dict을 사용하기 위한 key
    queries_key: str = "queries"
    most_recent_memories_token_key: str = "recent_memories_token"
    add_memory_key: str = "add_memory"
    relevant_memories_key: str = "relevant_memories"
    relevant_memories_simple_key: str = "relevant_memories_simple"
    most_recent_memories_key: str = "most_recent_memories"
    now_key: str = "now"
    chat_model: BaseChatModel

    def _chain(self, prompt: PromptTemplate) -> Runnable:
        return prompt | self.chat_model | StrOutputParser()

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        lines = re.split(r"\n", text.strip())
        lines = [line for line in lines if line.strip()]
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]
    
    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        content = []
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        created_time = memory.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
        return f"{prefix}[{created_time}] {memory.page_content.strip()}"

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    @property
    def memory_variables(self) -> List[str]:
        return []

    """
    _get_topics_of_reflection: 모든 메모리들을 취합해서 새로운 주제를 도출
    _get_insights_on_topic: 주제를 통해 새로운 영감 획득
    pause_to_reflect: 획득한 인사이트들을 memory에 추가
    """
    def _get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            "Given only the information above, what are the 3 most salient "
            "high-level questions we can answer about the subjects in the statements?\n"
            "Provide each question on a new line.")
        observations = self.working_memory[-last_k:]
        observation_str = "\n".join([self._format_memory_detail(o) for o in observations])
        result = self._chain(prompt).invoke({"observations": observation_str})
        return self._parse_list(result)
    
    def _get_insights_on_topic(self, topic: str, now: Optional[datetime] = None) -> List[str]:
        prompt = PromptTemplate.from_template(
            "Statements relevant to: '{topic}'\n"
            "---\n"
            "{related_statements}\n"
            "---\n"
            "What 5 high-level novel insights can you infer from the above statements "
            "that are relevant for answering the following question?\n"
            "Do not include any insights that are not relevant to the question.\n"
            "Do not repeat any insights that have already been made.\n\n"
            "Question: {topic}\n\n"
            "(example format: insight (because of 1, 5, 3))\n")
        related_memories = self.fetch_memories(topic, now=now)
        related_statements = "\n".join([self._format_memory_detail(memory, prefix=f"{i+1}. ") for i, memory in enumerate(related_memories)])
        result = self._chain(prompt).invoke({"topic": topic, "related_statements": related_statements})
        insights = self._parse_list(result)
        parsed_insights = []
        for insight in insights:
            match = re.match(r"(.*)\s+\(because of ([\d,\s]+)\)", insight)
            if match:
                insight_text = match.group(1).strip()
                memory_indices = [int(idx.strip()) - 1 for idx in match.group(2).split(",")]
                related_memories = [related_memories[idx] for idx in memory_indices]
                parsed_insights.append((insight_text, related_memories))
        return [insight[0] for insight in parsed_insights]

    def pause_to_reflect(self, now: Optional[datetime] = None) -> List[str]:
        new_insights = []
        topics = self._get_topics_of_reflection()
        
        for topic in topics:
            insights = self._get_insights_on_topic(topic, now=now)
            for insight in insights:
                self.add_memory(insight, now=now)
            new_insights.extend(insights)
        
        # Reflection 후 working_memory 비우기
        self.working_memory = []
        return new_insights
    
    """
    _score_memory_importance: 단일 메모리에 대하여 중요도 평가 (1~10)
    _score_memories_importance: batch로 기억 중요도 평가
    """

    def _score_memory_importance(self, memory_content: str) -> float:
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Respond with a single integer."
            + "\nMemory: {memory_content}"
            + "\nRating: ")
    
        score = self._chain(prompt).invoke({"memory_content": memory_content}).strip()
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def _score_memories_importance(self, memory_content: str) -> List[float]:
        prompt = PromptTemplate.from_template(
            "On the scale of 1 to 10, where 1 is purely mundane"
            + " (e.g., brushing teeth, making bed) and 10 is"
            + " extremely poignant (e.g., a break up, college"
            + " acceptance), rate the likely poignancy of the"
            + " following piece of memory. Always answer with only a list of numbers."
            + " If just given one memory still respond in a list."
            + " Memories are separated by semi colans (;)"
            + "\Memories: {memory_content}"
            + "\nRating: ")
        
        scores = self._chain(prompt).invoke({"memory_content": memory_content}).strip()
        scores_list = [float(x) for x in scores.split(";")]
        return scores_list

    """
    add_memories: batch로 vector DB에 add
    add_memory: 단일 memory를 vector db에 add
    """

    def add_memories(self, memory_content: str, now: datetime) -> List[str]:
        importance_scores = self._score_memories_importance(memory_content)
        self.aggregate_importance += max(importance_scores)
        memory_list = memory_content.split(";")
        
        documents = []
        for i in range(len(memory_list)):
            documents.append(Document(page_content=memory_list[i], metadata={"importance": importance_scores[i]}))
        
        # 단기 기억에 추가
        self.working_memory.extend(documents)
        print(self.working_memory)
        if (self.reflection_threshold is not None and self.aggregate_importance > self.reflection_threshold and not self.reflecting):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            self.aggregate_importance = 0.0
            self.reflecting = False

    def add_memory(self, memory_content: str, now: datetime) -> List[str]:
        importance_score = self._score_memory_importance(memory_content)
        self.aggregate_importance += importance_score
        document = Document(page_content=memory_content, metadata={"importance": importance_score})
        
        # 단기 기억에 추가
        self.working_memory.append(document)
        if (self.reflection_threshold is not None and self.aggregate_importance > self.reflection_threshold and not self.reflecting):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            self.aggregate_importance = 0.0
            self.reflecting = False
        print(self.working_memory)


    """
    fetch_memories: 메모리 검색 기능
    _get_memories_until_limit: token 제한까지 메모리 가져오기
    load_memory_variables: 들어온 dict에 따라 가변적으로 작동
    save_context: 메모리 저장 기능
    """

    def fetch_memories(self, observation: str, now: datetime) -> List[Document]:
        working_memories = [mem for mem in self.working_memory if observation in mem.page_content]
        long_term_memories = self.memory_retriever.invoke(input=observation, now=now)
        return working_memories + long_term_memories

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        result = []
        for doc in self.working_memory[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.chat_model.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc)
        return self.format_memories_simple(result)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        queries = inputs.get(self.queries_key)
        now = inputs.get(self.now_key)
        if queries is not None:
            relevant_memories = [mem for query in queries for mem in self.fetch_memories(query, now=now)]
            return {self.relevant_memories_key: self.format_memories_detail(relevant_memories),
                    self.relevant_memories_simple_key: self.format_memories_simple(relevant_memories)}
        most_recent_memories_token = inputs.get(self.most_recent_memories_token_key)
        if most_recent_memories_token is not None:
            return {self.most_recent_memories_key: self._get_memories_until_limit(most_recent_memories_token)}
        return {}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        mem = outputs.get(self.add_memory_key)
        now = outputs.get(self.now_key)
        if mem:
            self.add_memory(mem, now=now)

    def clear(self) -> None:
        self.working_memory = []

