"""
Database Schemas for Cleaning Service Booking App

Each Pydantic model below maps to a MongoDB collection with the lowercase
class name. Example: class User -> collection "user".

These schemas are used for validation on input and as a reference
for the database viewer.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal
from datetime import datetime

Role = Literal["customer", "cleaner", "admin"]
BookingStatus = Literal[
    "requested", "pending_acceptance", "scheduled",
    "on_the_way", "in_progress", "completed",
    "completed_paid", "refunded", "cancelled"
]
PaymentStatus = Literal["none", "authorized", "paid", "refunded", "failed"]
ServiceType = Literal["standard", "deep", "move_out"]

class Address(BaseModel):
    line1: str
    line2: Optional[str] = None
    city: str
    state: str
    postal_code: str

class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    role: Role = Field("customer", description="User role")
    password_hash: Optional[str] = Field(None, description="Hashed password")
    phone: Optional[str] = Field(None, description="Contact phone number")
    address: Optional[Address] = None
    stripe_customer_id: Optional[str] = Field(None, description="Stripe customer id")
    is_active: bool = Field(True, description="Whether user is active")

class CleanerProfile(BaseModel):
    user_id: str = Field(..., description="Reference to user _id")
    service_area_zip: List[str] = Field(default_factory=list, description="ZIP codes serviced")
    experience_years: Optional[int] = Field(0, ge=0)
    bio: Optional[str] = None
    rate_type: Literal["hourly", "flat"] = "hourly"
    hourly_rate: Optional[float] = Field(25.0, ge=0)
    flat_rate: Optional[float] = Field(None, ge=0)
    verified_document_paths: List[str] = Field(default_factory=list)
    rating_avg: float = 0.0
    rating_count: int = 0

class Service(BaseModel):
    name: ServiceType
    display_name: str
    base_price: float = Field(..., ge=0)
    hourly_multiplier: float = Field(1.0, ge=0)
    flat_multiplier: float = Field(1.0, ge=0)
    is_active: bool = True

class Booking(BaseModel):
    customer_id: str
    cleaner_id: Optional[str] = None
    service_type: ServiceType
    home_size: Optional[str] = Field(None, description="e.g., 1 bed/1 bath")
    duration_hours: Optional[float] = Field(None, ge=1, le=12)
    scheduled_start: datetime
    address: Address
    price_estimate: float = 0
    notes: Optional[str] = None
    status: BookingStatus = "pending_acceptance"
    payment_status: PaymentStatus = "none"
    payment_intent_id: Optional[str] = None
    invoice_id: Optional[str] = None

class Payment(BaseModel):
    booking_id: str
    customer_id: str
    amount: int = Field(..., description="Amount in cents")
    currency: str = "usd"
    status: PaymentStatus = "none"
    provider: str = "stripe"
    payment_intent_id: Optional[str] = None
    refund_id: Optional[str] = None

class Review(BaseModel):
    booking_id: str
    customer_id: str
    cleaner_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None

class Document(BaseModel):
    user_id: str
    filename: str
    path: str
    content_type: Optional[str] = None

# The schema endpoint will expose these for reference by tools.
