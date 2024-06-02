from django.contrib import admin
from django.urls import path
from AImodels.models import AImodel
from AImodels.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('predict', predict, name='predict'),
]
