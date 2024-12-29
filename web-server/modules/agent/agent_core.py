from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from typing import Any, Dict, List, Tuple
from datetime import datetime, timedelta
from .agent_memory import AgentMemory
import re, ast, json
        
class Agent(BaseModel):
    name : str # Agent의 이름
    age : int # Agent의 나이
    personality : str # Agent의 고유한 성격 
    status : Dict[str,Any] # Agent의 상태, 해당 프로젝트에서는 Energy, Health, Satisfaction으로 명시함. (Range : 1~10)
    friendship : Dict[str, Any] = Field(default_factory=dict) # 다른 Agent와의 호감도
    plan : Dict[str, Any] = Field(default_factory=dict) # Agent의 계획
    summary_refresh_seconds : int = 3600 # Agent의 현재 상태를 요약하는 주기 (Default : 1시간)
    memory : AgentMemory
    chat_model : BaseChatModel
    summary : str = "" # Agent의 요약
    last_refreshed : datetime = Field(default_factory=datetime.now) # 마지막 요약 시간

    class Config:
        arbitrary_types_allowed = True

    def _chain(self, prompt: PromptTemplate) -> Runnable:
        return prompt | self.chat_model | StrOutputParser()
    
    """
    _observation_get_entity : 관찰한 정보를 바탕으로 Entity 추출
    _observation_get_entity_action : 추출된 Entity가 관찰한 정보 내에서 어떤 행동을 하고 있는지 파악
    _observation : Agent와 Entity간의 관계와 Entity와 관련된 정보를 Memory로부터 load
    """

    def _observation_get_entity(self, observation: str) -> str:
        prompt = PromptTemplate.from_template("What is the observed entity in the following observation?" + "\n{observation}" + "\nEntity=")
        return self._chain(prompt).invoke({"observation" : observation}).strip()

    def _observation_get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template("What is the {entity_name} doing in the following observation?" +"\n{observation}" + "\nThe {entity_name} is")
        return self._chain(prompt).invoke({"entity_name" : entity_name, "observation" :observation}).strip()

    def _observation(self, observation: str, now : datetime) -> str:
        prompt = PromptTemplate.from_template(
            """
            {question}?
            Context from memory:
            {relevant_memories}
            Relevant context: 
            """)
        entity_name = self._observation_get_entity(observation)
        entity_action = self._observation_get_entity_action(observation, entity_name)
        inference_question = f"What is the relationship between {self.name} and {entity_name}"
        entity_describe = f"{entity_name} is {entity_action}"
        _relevant_memories = self.memory.load_memory_variables({self.memory.queries_key : [inference_question, entity_describe], self.memory.now_key : now})
        return self._chain(prompt).invoke({"question" : inference_question, "relevant_memories" : _relevant_memories}).strip()
    
    """
    _compute_agent_summary : 현재 Agent의 핵심 정보를 Memory부터 검색하여 load
    get_summary : 현재 시간을 바탕으로, 조건을 검토 후에 _compute_agent_summary를 호출하여 요약
    get_full_header : 현재 시간을 포함하여 이를 확인하기 쉽도록 parsing
    """

    def _compute_agent_summary(self, now : datetime) -> str:
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the following statements:"
            + "\n{relevant_memories}"
            + "\nDo not embellish."
            + "\nSummary: ")
        _relevant_memories = self.memory.load_memory_variables({self.memory.queries_key : [f"{self.name}'s core characteristics"], self.memory.now_key : now})
        return self._chain(prompt).invoke({"name" : self.name, "relevant_memories" :_relevant_memories}).strip()

    def get_summary(self, now :datetime, force_refresh: bool = False) -> str:
        current_time = now
        since_refresh = (current_time - self.last_refreshed).seconds
        if (not self.summary or since_refresh >= self.summary_refresh_seconds or force_refresh):
            self.summary = self._compute_agent_summary(now=now)
            print((f"Name: {self.name} (age: {self.age})"+ f"\nInnate traits: {self.personality}"+ f"\n{self.summary}\n"))
            self.last_refreshed = current_time
        return (f"Name: {self.name} (age: {self.age})"+ f"\nInnate traits: {self.personality}"+ f"\n{self.summary}")

    def get_full_header(self, now : datetime, force_refresh: bool = False) -> str:
        summary = self.get_summary(force_refresh=force_refresh, now=now)
        current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
        return (f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}")
        
    """
    _reaction : summary, current time, status, observation과 관련된 정보를 모두 취합하여 어떻게 반응할지를 결정
    reaction : _reaction에 prompt 형식을 넘겨주어, 관찰에 대한 반응을 결정(반응할지 대화할지를 결정)
    dialogue : _reaction에 prompt 형식을 넘겨주어, 대화를 진행(대화하는 것을 가정하고 진행)
    """

    def _reaction(self, observation: str, add_template: str, now: datetime) -> str:
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}" + "A status of 0 indicates the worst condition, while 10 indicates the best condition."
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n"
            + add_template)
        
        _agent_summary_description = self.get_summary(now=now)
        _relevant_memories_str = self._observation(observation, now=now)
        _current_time_str = now
        kwargs = {
            "agent_summary_description" : _agent_summary_description,
            "current_time" : _current_time_str,
            "relevant_memories" : _relevant_memories_str,
            "agent_name" : self.name,
            "observation" : observation,
            "agent_status" : self.status,
        }
        consumed_tokens = self.chat_model.get_num_tokens(prompt.format(most_recent_memories="", **kwargs))
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        kwargs["most_recent_memories"] = self.memory.load_memory_variables(kwargs)[self.memory.most_recent_memories_key]
        return self._chain(prompt=prompt).invoke(kwargs).strip()

    def reaction(self, observation: str, now: datetime) -> Tuple[bool, str]:
        reaction_template = (
            "Should {agent_name} react to the observation, and if so,"
            + " what would be an appropriate reaction? Respond in one line."
            + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n")
        full_result = self._reaction(observation, reaction_template, now=now)
        result = full_result.strip().split("\n")[0]
        self.memory.save_context({},{self.memory.add_memory_key: f"{self.name} observed "f"{observation} and reacted by {result}",self.memory.now_key: now,},)
        
        if "REACT:" in result:
            reaction = result.split("REACT:")[-1].strip()
            reaction = re.sub(f"^{self.name} ", "", reaction).strip()
            print(f"{self.name} : {reaction} \n")
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = result.split("SAY:")[-1].strip()
            said_value = re.sub(f"^{self.name} ", "", said_value).strip()
            print(f"{self.name} : {said_value} \n")
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def dialogue(self, observation: str, now: datetime, place : str) -> Tuple[bool, str]:
        dialogue_template = (
            "What would {agent_name} say in" + place + f"?Proceed with the conversation considering the status :"
            "To end the conversation, follow below format:"
            'GOODBYE: "what to say". Otherwise to continue the conversation,'
            'If you do not want to end the conversation, follow below format :'
            'write : SAY: "what to say next"\n'
            'Must Include "SAY:" or "GOODBYE:" in response \n')
        full_result = self._reaction(observation, dialogue_template, now=now)
        result = full_result.strip().split("\n")[0]

        if "GOODBYE:" in result:
            farewell = result.split("GOODBYE:")[-1].strip()
            farewell = re.sub(f"^{self.name} ", "", farewell).strip()
            self.memory.save_context({},{self.memory.add_memory_key: f"{self.name} observed " f"{observation} and said {farewell}", self.memory.now_key: now,},)
            print(f"{self.name} said {farewell}")
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = result.split("SAY:")[-1].strip()
            response_text = re.sub(f"^{self.name} ", "", response_text).strip()
            self.memory.save_context({},{self.memory.add_memory_key: f"{self.name} observed " f"{observation} and said {response_text}", self.memory.now_key: now,},)
            print(f"{self.name} said {response_text}")
            return True, f"{self.name} said {response_text}"
        else:
            return False, result
        
    """
    change_status : summary를 바탕으로 status를 변화
    make_daily_plan : 계획을 생성하기 위해, summary 및 약속과 관련된 기억을 바탕으로 새롭게 일일 plan을 수립
    make_event : 프로젝트 규모의 한계로, 게임 내의 모든 상호작용을 정의하기 어려워서 랜덤하게 해당 상황에 맞는 이벤트 발생.
    """

    def change_status(self, now : datetime) :
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}" 
            + "\nBased on the above information, update the agent's status considering current conditions."
            + "\nOn a scale of 0 to 10, where 0 represents a bad state and 10 represents a good state, rate the likely poignancy of the following piece of memory."
            + "\nRespond Only with a string in the format: 'Energy': x, 'Health': x, 'Satisfaction': x, where x is a single integer. Do not include any other information."
            + "\nRESULT : ")
        _agent_summary_description = self.get_summary(now=now)
        result = self._chain(prompt=prompt).invoke({"agent_summary_description": _agent_summary_description, "current_time": now, "agent_name": self.name, "agent_status": self.status}).strip()
        print(f"{now}, {self.name} : {result} \n")
        formatted_string = f"{{{result}}}"
        parsed_dict = ast.literal_eval(formatted_string)
        self.status = parsed_dict
        return parsed_dict
    
    def make_daily_plan(self, now : datetime) : 
        prompt = PromptTemplate.from_template(
        "{agent_summary_description}"
        +"\n {personality}"
        +"\n {relevant_memories}"
        +"\n Here are the places: Ethan's home, Jack's home, Oliver's home, Lilly's home, Emma's home, Mart, Park, Office, School."
        +"\n Based on the above information, Please give me a today schedule detailing where {agent_name} will be at every hour(0 ~ 24)."
        +"\n considering a typical daily routine if it is weekdays and weekends"
        +"\n Respond Only with a string in the format: '00:00' : '{agent_name}\'s home', '01:00' : '{agent_name}\'s home, ... , '08:00' : 'Park' "
        +"\n Do not include any other information. Plans are separated by comma(,)"
        +"\n RESULT : ")
        _agent_summary_description = self.get_summary(now=now)
        yesterday = now - timedelta(days=1)
        plan_question = f"Have any appointments or make any plans on In {yesterday}?"
        _relevant_memories = self.memory.load_memory_variables({self.memory.queries_key : [plan_question], self.memory.now_key : now})
        result = self._chain(prompt=prompt).invoke({"agent_summary_description": _agent_summary_description, "personality" : self.personality , "agent_name" : self.name, "relevant_memories" : _relevant_memories})
        result = result.replace("'", '"').replace('"s', "'s").replace('Ethan"s', "Ethan's")
        result = "{" + result + "}"
        plans_dict = json.loads(result)
        self.plan = plans_dict
        print(f"{now}, {self.name} : {self.plan} \n")
        

    def make_event(self, now : datetime) :
        prompt = PromptTemplate.from_template(
        """
        {agent_summary_description}
        Create only 1 random event considering {place} and {time} for {agent_name}. Prefer events related to objects rather than other people.
        Example: {agent_name} wakes up to the sound of a noisy construction site outside his window.
        Do not exaggerate or stray far from the given information.
        Just return the statement.
        EVENTS :
        """)
        _agent_summary_description = self.get_summary(now=now)
        result = self._chain(prompt=prompt).invoke({"agent_summary_description" : _agent_summary_description, "place" : self.plan[now.strftime("%H:%M")], "time" : now, "agent_name" : self.name})
        event_list = [x for x in result.split(";")]
        print(f"{now}, {self.name} : {event_list}\n")
        for event in event_list : 
            self.reaction(event, now)

    def npc_dialogue(self, partner_agent : str, chat_history : list, now: datetime, place : str) -> Tuple[bool, str]:
        dialogue_template = PromptTemplate.from_template(
            ("{agent_summary_description}"
            "\nIt is {current_time}. in {place}"
            "\n{agent_name}'s status: {agent_status}" + "A status of 0 indicates the worst condition, while 10 indicates the best condition."
            "\n{agent_name} has a friendship score of {friendship_score} towards {partner_agent}. A friendship score of 0 indicates the bad relationship, while 10 indicates the good relationship."
            "\n Followings are the context of the current conversation : {chat_history}"
            "\nSummary of relevant context from {agent_name}'s memory:"
            "\n{relevant_memories}"
            "To end the conversation, follow below format:"
            'GOODBYE: "what to say". Otherwise to continue the conversation,'
            'If you do not want to end the conversation, follow below format :'
            'write : SAY: "what to say next"\n'
            'Must Include "SAY:" or "GOODBYE:" in response \n'))
        _agent_summary_description = self.get_summary(now=now)

        kwargs = {
            "agent_summary_description" : _agent_summary_description,
            "current_time" : now,
            "place" : place,
            "agent_name" : self.name,
            "agent_status" : self.status,
            "friendship_score" : self.friendship[partner_agent],
            "partner_agent" : partner_agent,
            "relevant_memories" : self._observation(chat_history[-1], now=now)}
        full_result = self._chain(prompt=dialogue_template).invoke(kwargs).strip()
        result = full_result.strip().split("\n")[0]

        if "GOODBYE:" in result:
            farewell = result.split("GOODBYE:")[-1].strip()
            farewell = re.sub(f"^{self.name} ", "", farewell).strip()
            self.memory.save_context({},{self.memory.add_memory_key: f"{self.name} observed " f"{chat_history[-1]} and said {farewell}", self.memory.now_key: now,},)
            print(f"{self.name} said {farewell}")
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = result.split("SAY:")[-1].strip()
            response_text = re.sub(f"^{self.name} ", "", response_text).strip()
            self.memory.save_context({},{self.memory.add_memory_key: f"{self.name} observed " f"{chat_history[-1]} and said {response_text}", self.memory.now_key: now,},)
            print(f"{self.name} said {response_text}")
            return True, f"{self.name} said {response_text}"
        else:
            return False, result
        
    def calc_friendship(self, partner_agent : str, chat_history : list, now : datetime) :
        prompt = PromptTemplate.from_template("Summarize the key points from the conversation below from {self_name}'s perspective,\
                                              and separate each summary with a semicolon.\
                                              CHAT_HISTORY : {chat_history}")
        result = self._chain(prompt).invoke({"self_name" : self.name, "chat_history" : chat_history}).strip()
        feature_list = [x for x in result.split(";")]
        score = list.pop()
        for keypoint in feature_list :
            self._observation(keypoint, now)
        total_score = self.friendship[partner_agent] + score
        self.friendship[partner_agent] = total_score
        