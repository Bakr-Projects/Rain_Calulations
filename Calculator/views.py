from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.urls import reverse
from django.contrib import messages
from .Views import RainIntensity, TimeOfConcentrationView,AboutUs

def TC(request):
    return TimeOfConcentrationView.TC(request)  

def RI(request):
    return RainIntensity.RI(request)
    
def About(request):
    return AboutUs.About(request)