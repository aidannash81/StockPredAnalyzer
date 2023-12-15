from django.urls import path
from . import views

urlpatterns = [
    path('', views.stock_view, name='stock-view'),
    path('loading/', views.loading_view, name='loading-view'),
    path('result/', views.result_view, name='result-view')
]