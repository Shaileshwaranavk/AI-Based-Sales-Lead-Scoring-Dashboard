from django.urls import path
from . import views

urlpatterns = [
    # === Lead Scoring & Analysis ===
    path("analyze-leads/", views.analyze_leads, name="analyze_leads"),

    # === Sales Pitch Generation ===
    path("generate-sales-pitch/", views.generate_sales_pitch, name="generate_sales_pitch"),

    # === CRM System ===
    path("crm/add-customer/", views.crm_add_customer, name="crm_add_customer"),
    path("crm/add-review/", views.crm_add_review, name="crm_add_review"),
    path("crm/retrain-models/", views.crm_retrain_models, name="crm_retrain_models"),
    path("crm/get-customers/", views.crm_get_customers, name="crm_get_customers"),  # ✅ NEW
]
