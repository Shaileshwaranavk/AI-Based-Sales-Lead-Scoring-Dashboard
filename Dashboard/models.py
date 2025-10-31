from django.db import models

class Customer(models.Model):
    customer_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    company = models.CharField(max_length=255, blank=True, null=True)
    industry = models.CharField(max_length=255, blank=True, null=True)
    country = models.CharField(max_length=100, blank=True, null=True)
    product_interested = models.CharField(max_length=255, blank=True, null=True)
    revenue_potential = models.FloatField(default=0)
    conversion_rate = models.FloatField(default=0)  # target label for training

    def __str__(self):
        return f"{self.name} ({self.email})"

class CustomerReview(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="reviews")
    review_text = models.TextField()
    sentiment_score = models.FloatField(default=0.0)  # computed from AI
    date_added = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review for {self.customer.name} ({self.sentiment_score})"
