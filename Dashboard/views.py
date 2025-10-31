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
