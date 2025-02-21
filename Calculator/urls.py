from django.urls import path
from . import views
from Calculator import views

urlpatterns = [
    path('',views.TC, name='TC'),
    path('TC',views.TC,name='TC'),
    path('RI',views.RI,name='RI'),
    path('About',views.About,name='About'),
    
]
