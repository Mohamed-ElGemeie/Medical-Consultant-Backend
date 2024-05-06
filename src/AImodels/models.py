from django.db import models

class AImodel(models.Model):
    name = models.TextField(default="")
    h5 = models.TextField(default="")