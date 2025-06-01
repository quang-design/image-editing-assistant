"""
Shared data models for the image editing assistant
"""

from typing import List, Optional, Union
from pydantic import BaseModel
from logic.router_agent import ActionType

# Define structured response models for each agent type
class ImageMetadata(BaseModel):
    width: int
    height: int
    format: str
    color_space: str
    channels: int
    bit_depth: int

class HistogramData(BaseModel):
    red: List[int]
    green: List[int]
    blue: List[int]
    luminance: List[int]

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float

class InfoResponse(BaseModel):
    metadata: ImageMetadata
    histogram: HistogramData
    dominant_colors: List[str]
    description: str

class EditResponse(BaseModel):
    edited_image_path: str
    edits_applied: List[str]
    message: str

class LocalEditResponse(BaseModel):
    edited_image_path: str
    detected_objects: List[BoundingBox]
    edited_regions: List[BoundingBox]
    message: str

class ClarifyResponse(BaseModel):
    message: str
    suggested_prompts: List[str]

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

class AssistantResponse(BaseModel):
    action: ActionType
    info_data: Optional[InfoResponse] = None
    edit_data: Optional[Union[EditResponse, LocalEditResponse]] = None
    clarify_data: Optional[ClarifyResponse] = None
    error: Optional[ErrorResponse] = None
