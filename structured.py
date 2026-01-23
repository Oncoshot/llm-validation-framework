from pydantic import BaseModel
from typing import Any, List, Dict, Optional, Union
from shared.version_info import version_string

# For all LLM Extracted Value
# Multi Value: Lists -> Stringified lists'[A, B]' (no quotes if use remove_quotes = True)
# Single Value String -> String 
# None or No Date-> None (after applied to_sortable_date)

class StructuredField(BaseModel):
    name: str
    value: Any
    justification: Optional[str] = None
    confidence: Optional[str] = None

class StructuredGroup(BaseModel):
    group_name: str
    fields: List[StructuredField]

class StructuredResult(BaseModel):
    version: Optional[str] = version_string()
    batch: Optional[str] = None
    groups: List[StructuredGroup]


