from django.shortcuts import render

class AboutUs:
    def __init__(self):
        pass
    def handle_request(self, request):
        return render(request, "About.html")
        
    
    
def About(request):
    view = AboutUs()
    return view.handle_request(request)

