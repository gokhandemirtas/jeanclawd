from dataclasses import dataclass


@dataclass
class SystemMessage:
	content: str


@dataclass
class HumanMessage:
	content: str


@dataclass
class AIMessage:
	content: str
