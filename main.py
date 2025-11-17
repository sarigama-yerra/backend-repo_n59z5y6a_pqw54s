import os
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from database import db, create_document, get_documents
from schemas import (
    User, CleanerProfile, Service, Booking, Payment, Review,
)

# Stripe (mocked by default if key missing)
import requests

app = FastAPI(title="Cleaning Service Booking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STRIPE_SECRET = os.getenv("STRIPE_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Helpers

def collection(name: str):
    return db[name]


def stripe_available() -> bool:
    return bool(STRIPE_SECRET)


# Utility response models
class AuthRequest(BaseModel):
    name: Optional[str] = None
    email: EmailStr
    password: Optional[str] = None
    role: Optional[str] = "customer"


class AuthResponse(BaseModel):
    user_id: str
    email: EmailStr
    role: str


class CardSetupRequest(BaseModel):
    user_id: str


class BookingRequest(BaseModel):
    customer_id: str
    service_type: str
    scheduled_start: datetime
    duration_hours: Optional[float] = None
    home_size: Optional[str] = None
    address_line1: str
    address_line2: Optional[str] = None
    city: str
    state: str
    postal_code: str
    notes: Optional[str] = None


class AcceptBookingRequest(BaseModel):
    cleaner_id: str


class StatusUpdateRequest(BaseModel):
    status: str


class RefundRequest(BaseModel):
    reason: Optional[str] = None


@app.get("/")
def root():
    return {"message": "Cleaning Service API is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "❌ Not Set",
        "database_name": "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()
        return response
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
        return response


# ------------------------ AUTH ------------------------
@app.post("/api/auth/signup", response_model=AuthResponse)
def signup(payload: AuthRequest):
    # Very simple mock auth for demo: check if email exists.
    existing = collection("user").find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        name=payload.name or payload.email.split("@")[0],
        email=payload.email,
        role=(payload.role or "customer"),
        password_hash=payload.password or None,
    )
    user_id = create_document("user", user)
    # Create Stripe customer if available
    stripe_customer_id = None
    if stripe_available():
        try:
            r = requests.post(
                "https://api.stripe.com/v1/customers",
                headers={"Authorization": f"Bearer {STRIPE_SECRET}"},
                data={"email": user.email, "name": user.name},
                timeout=10,
            )
            if r.status_code == 200:
                stripe_customer_id = r.json().get("id")
                collection("user").update_one({"_id": collection("user").find_one({"email": user.email})["_id"]}, {"$set": {"stripe_customer_id": stripe_customer_id}})
        except Exception:
            pass
    return AuthResponse(user_id=str(user_id), email=user.email, role=user.role)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: AuthRequest):
    doc = collection("user").find_one({"email": payload.email})
    if not doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return AuthResponse(user_id=str(doc["_id"]), email=doc["email"], role=doc.get("role", "customer"))


# ------------------------ SERVICES ------------------------
@app.get("/api/services", response_model=List[Service])
def list_services():
    items = get_documents("service")
    return [Service(**{k: v for k, v in item.items() if k not in ["_id", "created_at", "updated_at"]}) for item in items]


@app.post("/api/services", response_model=dict)
def create_service(service: Service):
    sid = create_document("service", service)
    return {"id": sid}


# ------------------------ CLEANER PROFILE ------------------------
@app.post("/api/cleaners/profile", response_model=dict)
def upsert_cleaner_profile(profile: CleanerProfile):
    existing = collection("cleanerprofile").find_one({"user_id": profile.user_id})
    if existing:
        collection("cleanerprofile").update_one({"_id": existing["_id"]}, {"$set": profile.model_dump()})
        return {"id": str(existing["_id"]) }
    else:
        cid = create_document("cleanerprofile", profile)
        return {"id": cid}


@app.post("/api/cleaners/upload-id")
def upload_id(user_id: str = Form(...), file: UploadFile = File(...)):
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    # Store a document record
    doc = {
        "user_id": user_id,
        "filename": file.filename,
        "path": file_path,
        "content_type": file.content_type,
        "created_at": datetime.utcnow(),
    }
    collection("document").insert_one(doc)
    return {"message": "Uploaded", "path": file_path}


# ------------------------ BOOKING ------------------------

def estimate_price(service_type: str, duration_hours: Optional[float], cleaner_rate: Optional[float] = None, base_price: Optional[float] = None) -> float:
    svc = collection("service").find_one({"name": service_type})
    base = base_price if base_price is not None else (svc.get("base_price") if svc else 50)
    rate = cleaner_rate or 25
    if duration_hours:
        return round(base + rate * duration_hours, 2)
    return float(base)


@app.post("/api/bookings/estimate")
def booking_estimate(payload: BookingRequest):
    # Find any sample cleaner rate in area if exists
    cleaner = collection("cleanerprofile").find_one({"service_area_zip": {"$in": [payload.postal_code]}})
    price = estimate_price(payload.service_type, payload.duration_hours, cleaner_rate=(cleaner or {}).get("hourly_rate"))
    return {"price_estimate": price}


@app.post("/api/bookings", response_model=dict)
def create_booking(payload: BookingRequest):
    # Match cleaners in area
    cleaner = collection("cleanerprofile").find_one({"service_area_zip": {"$in": [payload.postal_code]}})
    price = estimate_price(payload.service_type, payload.duration_hours, cleaner_rate=(cleaner or {}).get("hourly_rate"))

    booking = Booking(
        customer_id=payload.customer_id,
        cleaner_id=str(cleaner["user_id"]) if cleaner else None,
        service_type=payload.service_type,        
        scheduled_start=payload.scheduled_start,
        duration_hours=payload.duration_hours,
        home_size=payload.home_size,
        address={
            "line1": payload.address_line1,
            "line2": payload.address_line2,
            "city": payload.city,
            "state": payload.state,
            "postal_code": payload.postal_code,
        },
        notes=payload.notes,
        price_estimate=price,
        status="pending_acceptance" if cleaner else "requested",
    )
    bid = create_document("booking", booking)

    # Create or attach payment method (setup intent) if Stripe available
    payment_intent_id = None
    if stripe_available():
        try:
            # create a SetupIntent so customer can add a card; for demo we simulate id
            payment_intent_id = f"seti_{bid}"
        except Exception:
            pass
    if payment_intent_id:
        collection("booking").update_one({"_id": collection("booking").find_one({"_id": collection("booking").find_one({"_id": collection("booking").find_one({"_id": None})})})}, {"$set": {"payment_intent_id": payment_intent_id}})
    return {"id": bid, "price_estimate": price, "status": booking.status}


@app.post("/api/bookings/{booking_id}/accept")
def accept_booking(booking_id: str, payload: AcceptBookingRequest):
    booking = collection("booking").find_one({"_id": collection("booking").find_one({"_id": None})})
    # In this environment, we can't reference ObjectId easily; use string ids returned by create_document
    collection("booking").update_one({"_id": booking_id}, {"$set": {"cleaner_id": payload.cleaner_id, "status": "scheduled"}})
    return {"message": "Accepted"}


@app.post("/api/bookings/{booking_id}/status")
def update_status(booking_id: str, payload: StatusUpdateRequest):
    valid = {"scheduled", "on_the_way", "in_progress", "completed"}
    if payload.status not in valid:
        raise HTTPException(status_code=400, detail="Invalid status")
    collection("booking").update_one({"_id": booking_id}, {"$set": {"status": payload.status}})

    # On completion trigger payment
    if payload.status == "completed":
        _charge_and_complete(booking_id)
    return {"message": "Updated"}


def _charge_and_complete(booking_id: str):
    booking = collection("booking").find_one({"_id": booking_id})
    if not booking:
        return
    amount_cents = int(float(booking.get("price_estimate", 0)) * 100)

    paid = False
    payment_intent_id = booking.get("payment_intent_id")
    if stripe_available() and amount_cents > 0:
        # For demo: simulate charge
        paid = True
        payment_intent_id = payment_intent_id or f"pi_{booking_id}"
    else:
        paid = True  # simulate success in demo

    if paid:
        collection("booking").update_one({"_id": booking_id}, {"$set": {"status": "completed", "payment_status": "paid"}})
        create_document("payment", {
            "booking_id": booking_id,
            "customer_id": booking.get("customer_id"),
            "amount": amount_cents,
            "currency": "usd",
            "status": "paid",
            "provider": "stripe" if stripe_available() else "mock",
            "payment_intent_id": payment_intent_id,
        })


@app.post("/api/bookings/{booking_id}/refund")
def refund_booking(booking_id: str, payload: RefundRequest):
    booking = collection("booking").find_one({"_id": booking_id})
    if not booking:
        raise HTTPException(status_code=404, detail="Booking not found")
    # For demo, simply mark refunded
    collection("booking").update_one({"_id": booking_id}, {"$set": {"payment_status": "refunded", "status": "refunded"}})
    create_document("payment", {
        "booking_id": booking_id,
        "customer_id": booking.get("customer_id"),
        "amount": int(float(booking.get("price_estimate", 0)) * 100) * -1,
        "currency": "usd",
        "status": "refunded",
        "provider": "stripe" if stripe_available() else "mock",
    })
    return {"message": "Refunded"}


# ------------------------ LIST & DASHBOARDS ------------------------
@app.get("/api/customers/{user_id}/bookings")
def list_customer_bookings(user_id: str):
    items = get_documents("booking", {"customer_id": user_id})
    return [{**i, "id": str(i.get("_id"))} for i in items]


@app.get("/api/cleaners/{user_id}/bookings")
def list_cleaner_bookings(user_id: str):
    items = get_documents("booking", {"cleaner_id": user_id})
    return [{**i, "id": str(i.get("_id"))} for i in items]


@app.get("/api/admin/metrics")
def admin_metrics():
    total_bookings = collection("booking").count_documents({})
    revenue_paid = sum([p.get("amount", 0) for p in get_documents("payment", {"status": "paid"})]) / 100.0
    active_cleaners = collection("cleanerprofile").count_documents({})
    users = collection("user").count_documents({})
    return {
        "total_bookings": total_bookings,
        "revenue_paid": revenue_paid,
        "active_cleaners": active_cleaners,
        "users": users,
    }


@app.post("/api/reviews", response_model=dict)
def create_review(review: Review):
    rid = create_document("review", review)
    # update cleaner rating summary
    cleaner_id = review.cleaner_id
    reviews = get_documents("review", {"cleaner_id": cleaner_id})
    if reviews:
        avg = sum([r.get("rating", 0) for r in reviews]) / len(reviews)
        collection("cleanerprofile").update_one({"user_id": cleaner_id}, {"$set": {"rating_avg": avg, "rating_count": len(reviews)}})
    return {"id": rid}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
