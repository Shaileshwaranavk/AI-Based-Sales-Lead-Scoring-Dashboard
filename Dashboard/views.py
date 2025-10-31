from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .model_pipeline import run_hybrid_pipeline
import os, json

@csrf_exempt
def analyze_leads(request):
    """
    API endpoint to process lead CSVs and return hybrid scores.
    Accepts:
    - POST request with:
        - new_data (CSV file)
        - labeled_data (optional CSV file)
        - product_name, description, features, api_key (strings)
    """
    if request.method != "POST":
        return JsonResponse({"error": "Use POST request"}, status=400)

    try:
        product_name = request.POST.get("product_name", "Generic Product")
        description = request.POST.get("description", "")
        features = request.POST.get("features", "")

        new_file = request.FILES.get("new_data")
        if not new_file:
            return JsonResponse({"error": "Missing new_data CSV file"}, status=400)

        new_path = f"media/{new_file.name}"
        os.makedirs("media", exist_ok=True)
        with open(new_path, "wb+") as dest:
            for chunk in new_file.chunks():
                dest.write(chunk)

        labeled_path = None
        labeled_file = request.FILES.get("labeled_data")
        if labeled_file:
            labeled_path = f"media/{labeled_file.name}"
            with open(labeled_path, "wb+") as dest:
                for chunk in labeled_file.chunks():
                    dest.write(chunk)

        result = run_hybrid_pipeline(new_path, labeled_path, product_name, description, features, top_n=10)

        return JsonResponse(result, safe=False)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .model_pipeline import run_hybrid_pipeline
from Dashboard.Sales_pitch.infer import generate_recommendation  # ✅ import hybrid sales generator
import json

@csrf_exempt
def generate_sales_pitch(request):
    """
    API endpoint to generate AI-based sales recommendations.
    Accepts:
    - POST request with:
        - product_name (string)
        - description (string)
        - features (comma-separated string)
    Returns:
        - Feature weightage
        - Highlighted features
        - LLM-generated sales pitch
    """
    if request.method != "POST":
        return JsonResponse({"error": "Use POST request"}, status=400)

    try:
        data = json.loads(request.body.decode("utf-8"))
        product_name = data.get("product_name", "")
        description = data.get("description", "")
        features = data.get("features", "")

        if not product_name or not description or not features:
            return JsonResponse({"error": "Missing product_name, description, or features"}, status=400)

        result = generate_recommendation(product_name, description, features)
        return JsonResponse(result, status=200)

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from Dashboard.CRM.crm_trainer import retrain_models
from Dashboard.Sales_pitch.infer import generate_recommendation
from Dashboard.model_pipeline import run_hybrid_pipeline
from Dashboard.models import Customer, CustomerReview
import json

@csrf_exempt
def crm_add_customer(request):
    if request.method != "POST":
        return JsonResponse({"error": "Use POST request"}, status=400)
    try:
        data = json.loads(request.body.decode("utf-8"))
        c = Customer.objects.create(
            name=data["name"],
            email=data["email"],
            company=data.get("company", ""),
            industry=data.get("industry", ""),
            country=data.get("country", ""),
            product_interested=data.get("product_interested", ""),
            revenue_potential=data.get("revenue_potential", 0),
            conversion_rate=data.get("conversion_rate", 0)
        )
        return JsonResponse({"status": "✅ Customer added", "customer_id": c.customer_id})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def crm_add_review(request):
    if request.method != "POST":
        return JsonResponse({"error": "Use POST request"}, status=400)
    try:
        data = json.loads(request.body.decode("utf-8"))
        customer = Customer.objects.get(customer_id=data["customer_id"])
        review = CustomerReview.objects.create(customer=customer, review_text=data["review"])
        return JsonResponse({"status": "✅ Review added", "review_id": review.id})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from Dashboard.CRM.crm_trainer import retrain_models

logger = logging.getLogger(__name__)

@csrf_exempt
def crm_retrain_models(request):
    """
    API endpoint to retrain both ML and LLM models using customer CRM data.

    ✅ Method: POST
    🔧 Process:
        1. Collects customer and review data from the database.
        2. Creates a dataset (crm_training_data.csv).
        3. Retrains:
            - RandomForest model (lead prediction)
            - Fine-tuned T5 model (sales pitch)
    Returns:
        JSON with success or error message.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Use POST request"}, status=400)

    try:
        logger.info("Initiating CRM model retraining...")
        result = retrain_models()
        logger.info("CRM model retraining completed successfully.")
        return JsonResponse({"status": result}, status=200)
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)



from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from Dashboard.models import Customer, CustomerReview

@csrf_exempt
def crm_get_customers(request):
    """
    GET endpoint to fetch all customers or a single customer (by ID).
    Optional query parameter: ?customer_id=<id>
    """
    if request.method != "GET":
        return JsonResponse({"error": "Use GET request"}, status=400)

    try:
        customer_id = request.GET.get("customer_id", None)

        if customer_id:
            # ✅ Fetch one customer + its reviews
            customer = Customer.objects.get(customer_id=customer_id)
            reviews = list(CustomerReview.objects.filter(customer=customer).values("id", "review_text", "created_at"))
            data = {
                "customer_id": customer.customer_id,
                "name": customer.name,
                "email": customer.email,
                "company": customer.company,
                "industry": customer.industry,
                "country": customer.country,
                "product_interested": customer.product_interested,
                "revenue_potential": customer.revenue_potential,
                "conversion_rate": customer.conversion_rate,
                "reviews": reviews
            }
            return JsonResponse(data, status=200)

        else:
            # ✅ Fetch all customers (no reviews to keep lightweight)
            customers = list(Customer.objects.all().values(
                "customer_id", "name", "email", "company", "industry",
                "country", "product_interested", "revenue_potential", "conversion_rate"
            ))
            return JsonResponse({"customers": customers, "count": len(customers)}, status=200)

    except Customer.DoesNotExist:
        return JsonResponse({"error": "Customer not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
