from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal
import cohere
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import csv
from io import StringIO
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import random

# Load environment variables
env_path = Path('.') / 'variables.env'
load_dotenv(dotenv_path=env_path)

# Input Models with new fields for A/B testing
class Contact(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    job_title: str = Field(..., min_length=1, max_length=100)
    group: Literal["A", "B"] = "A"  # Default group for A/B testing

class Account(BaseModel):
    account_name: str = Field(..., min_length=1, max_length=200)
    industry: str = Field(..., min_length=1, max_length=100)
    pain_points: List[str] = Field(..., min_items=1, max_items=5)
    contacts: List[Contact] = Field(..., min_items=1)
    campaign_objective: Literal["awareness", "nurturing", "upselling"]

    # New fields for interest, tone, and language
    interest: Literal["high", "medium", "low"] = "medium"  # Level of interest
    tone: Literal["formal", "casual", "enthusiastic", "neutral"] = "neutral"  # Tone of the email
    language: str = Field(..., min_length=1, max_length=200)  # Language for the email

class EmailVariant(BaseModel):
    subject: str
    body: str
    call_to_action: str

class Email(BaseModel):
    variants: List[EmailVariant]

class Campaign(BaseModel):
    account_name: str
    emails: List[Email]

class CampaignRequest(BaseModel):
    accounts: List[Account] = Field(..., min_items=1, max_items=10)
    number_of_emails: int = Field(..., gt=0, le=10)

class CampaignResponse(BaseModel):
    campaigns: List[Campaign]

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate environment on startup
    if not os.getenv("COHERE_API_KEY"):
        raise ValueError("COHERE_API_KEY environment variable is not set")
    yield

app = FastAPI(
    title="Email Drip Campaign API with A/B Testing",
    description="Generate personalized email campaigns with A/B testing using Cohere",
    version="1.0.0",
    lifespan=lifespan
)

# Dependency for Cohere client
def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Cohere API key not found")
    return cohere.Client(api_key)

def generate_email_content(client: cohere.Client, account: Account, email_number: int, total_emails: int) -> List[EmailVariant]:
    """Generate multiple email variants for A/B testing."""
    variants = []
    for tone in ["formal", "casual"]:  # Example tones for A/B testing
        prompt = f"""
        Create a personalized email for the following business account:
        Company: {account.account_name}
        Industry: {account.industry}
        Pain Points: {', '.join(account.pain_points)}
        Campaign Stage: Email {email_number} of {total_emails}
        Campaign Objective: {account.campaign_objective}
        Recipient Job Title: {account.contacts[0].job_title}

        Interest: {account.interest}
        Tone: {tone}
        Language: {account.language}

        Generate a JSON response with:
        1. An engaging and catchy subject line
        2. Personalized email body
        3. Clear call-to-action

        Format the response as valid JSON with keys: "subject", "body", "call_to_action"
        """
        try:
            response = client.generate(
                model="command-xlarge-nightly",
                prompt=prompt,
                max_tokens=300,
                temperature=0.7,
            )
            email_data = eval(response.generations[0].text.strip())
            variants.append(
                EmailVariant(
                    subject=email_data["subject"],
                    body=email_data["body"],
                    call_to_action=email_data["call_to_action"]
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating email variant: {str(e)}"
            )
    return variants

def generate_campaign(client: cohere.Client, account: Account, number_of_emails: int) -> Campaign:
    """Generate a complete email campaign with A/B testing variants."""
    try:
        emails = []
        for contact in account.contacts:
            # Randomly assign groups for A/B testing
            contact.group = random.choice(["A", "B"])
        
        for i in range(number_of_emails):
            email_variants = generate_email_content(client, account, i + 1, number_of_emails)
            emails.append(Email(variants=email_variants))
        
        return Campaign(account_name=account.account_name, emails=emails)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating campaign for {account.account_name}: {str(e)}"
        )

@app.post(
    "/generate-campaigns/",
    response_model=CampaignResponse,
    summary="Generate email campaigns with A/B testing",
    response_description="Generated email campaigns for the provided accounts"
)
def generate_campaigns(
    request: CampaignRequest,
    client: cohere.Client = Depends(get_cohere_client)
) -> CampaignResponse:
    """Generate personalized email campaigns for multiple accounts."""
    try:
        campaigns = []
        for account in request.accounts:
            campaign = generate_campaign(client, account, request.number_of_emails)
            campaigns.append(campaign)
        
        return CampaignResponse(campaigns=campaigns)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating campaigns: {str(e)}"
        )

@app.post(
    "/export-campaigns-csv/",
    summary="Export campaigns as CSV",
    response_description="CSV file containing all generated campaigns"
)
def export_campaigns_csv(
    request: CampaignRequest,
    client: cohere.Client = Depends(get_cohere_client)
):
    """Export campaigns in CSV format for email automation tools."""
    try:
        # Generate the campaigns first
        campaigns_response = generate_campaigns(request, client)
        
        # Create CSV in memory
        output = StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Account Name', 'Email Number', 'Variant', 'Subject', 'Body', 'Call to Action'])
        
        # Write data
        for campaign in campaigns_response.campaigns:
            for i, email in enumerate(campaign.emails, 1):
                for variant_idx, variant in enumerate(email.variants, 1):
                    writer.writerow([
                        campaign.account_name,
                        f"Email {i}",
                        f"Variant {variant_idx}",
                        variant.subject,
                        variant.body,
                        variant.call_to_action
                    ])
        
        # Prepare the response
        output.seek(0)
        filename = f"campaigns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "text/csv"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting campaigns to CSV: {str(e)}"
        )

@app.get(
    "/health",
    summary="Health check endpoint",
    response_description="Current API health status"
)
def health_check():
    """Health check endpoint to verify API status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "cohere_api_configured": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
