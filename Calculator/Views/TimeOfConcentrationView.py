from django.shortcuts import render
from django.contrib import messages

class TimeOfConcentrationView:
    """
    Class-based implementation for handling the 'Time of Concentration' page.
    """

    # List of all formulas
    FORMULAS_NAMES = [
        "Bransby-Williams’s Model",
        "Giandotti’s Model",
        "Kirpich’s Model",
        "Johnstone and Cross’s Model",
        "California Culvert Practice's Methodology",
        "Federal Aviation Agency’s Formula",
        "NRCS-SCS method",
        "Texas Department of transportation’s formula",
        "Ventura’s Formula",
        "Pasini’s Formula",
        "Tournon’s Formula",
        "Puglisi and Zanframundo’s Formula",
        "Fattorelli and Marchi's Formula",
        "Pezzoli's Formula",
        "Eaton's Method",
        "Sheridan's Formula",
        "Témez's Formula",
        "Chow's Model",
        "Dooge's Formula",
        "Hatkanir and Sezen",
        "Espey's Method",
        "Pilgrim and McDermott's Formula",
        "Ferro's Model",
        "Bocchiola's Model",
        "AddManually",
    ]

    def __init__(self):
        # Default input values and context
        self.default_data = {
            "H": "",
            "L": "",
            "S": "",
            "A": "",
            "DeltaH": "",
            "Slope_mean": "",
            "r": "",
            "C": "",
            "CN": "",
        }
        self.context = {
            "data": self.default_data.copy(),
            "formulas_names": self.FORMULAS_NAMES,
            "selected_formulas_names": [],
            "averageTC": 0,
            "addManully": 0
        }

    def handle_request(self, request):
        """
        Handles the HTTP request for the Time of Concentration page.
        """
        if request.method == "POST":
            action = request.POST.get("action")

            if action == "calculate":
                self.context = self.calculate(request)
            
        return render(request, "TC.html", self.context)

    def calculate(self, request):
        """
        Handles the calculation of average TC based on selected formulas and user input.
        """        
        addManully = 0
        AddM = request.POST.get("AddManually", None)
        selected_formulas_names = request.POST.getlist("formula")
        if AddM is not None:
            selected_formulas_names.append("AddManually")
            addManully = request.POST.get("addManully", 0)

        data = self._parse_numeric_inputs(request.POST)
        if data is None:
            messages.warning(request,'error: Please enter valid numeric values.')
        if selected_formulas_names is None:
            messages.warning(request,'error: Please choose at least one formulas.')
        formulas_equations = self._get_formulas_equations(data)
        
        formulas_equations.update({'AddManually':float(addManully)})

        selected_values = [
            formulas_equations[name]
            for name in selected_formulas_names
                if name in formulas_equations
            ]
        print(f"here: {selected_values}")
        averageTC = sum(selected_values) / len(selected_values) if selected_values else 0
        # addManully=''
        # else:
        #     selected_formulas_names = []
        #     data = []
        #     averageTC = request.POST.get("addManully", 0)
        #     addManully = averageTC

        return {
            "data": data,
            "formulas_names": self.FORMULAS_NAMES,
            "selected_formulas_names": selected_formulas_names,
            "averageTC": round(float(averageTC), 4),
            "addManully": addManully,
        }

    def _parse_numeric_inputs(self, post_data):
        """
        Parses numeric inputs from the POST data with default values for invalid inputs.
        """
        try:
            return {
                "H": float(post_data.get("H", 0)),
                "L": float(post_data.get("L", 0)),
                "S": float(post_data.get("S", 0)),
                "A": float(post_data.get("A", 0)),
                "DeltaH": float(post_data.get("DH", 0)),
                "Slope_mean": float(post_data.get("Slope_mean", 0)),
                "r": float(post_data.get("r", 0)),
                "C": float(post_data.get("C", 0)),
                "CN": float(post_data.get("CN", 0)),
            }
        except ValueError:
            print("error: Please enter valid numeric values.")
            return None

    def _get_formulas_equations(self, data):
        """
        Returns a dictionary of hydrological formulas and their calculated values.
        Includes error handling and input validation.
    
        Parameters:
            data (dict): A dictionary containing required hydrological parameters.
    
        Returns:
            dict: Formula names as keys and their computed values as values.
        """
        try:
            # Required keys for calculations
            required_keys = ['L', 'A', 'S', 'H', 'DeltaH', 'C', 'Slope_mean', 'CN', 'r']
        
            # Validate that all required keys are in the data
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise KeyError(f"Missing required data keys: {missing_keys}")

            eps = 1e-6  # Small epsilon to avoid division by zero

            # Precompute common expressions for efficiency
            L, A, S = data['L']/1000, data['A'], data['S'] + eps
            H, DeltaH, C = data['H'] + eps, data['DeltaH'] + eps, data['C']
            Slope_mean, CN, r = data['Slope_mean'] + eps, data['CN'] + eps, data['r'] + eps

            # Compute and return formulas
            formulas = {
                "Bransby-Williams’s Model": 0.2426 * L * A ** -0.1 * S ** -0.2,
                "Giandotti’s Model": (4 * A ** 0.5 + 1.5 * L) / (0.8 * (H) ** 0.5),
                "Kirpich’s Model": 0.0663 * L ** 0.77 * S ** -0.385,
                "Johnstone and Cross’s Model": 0.0543 * L ** 0.5 * S ** -0.5,
                "California Culvert Practice's Methodology": 0.95 * L ** 1.155 * (DeltaH) ** -0.385,
                "Federal Aviation Agency’s Formula": 0.3788 * (1.1 - C) * L ** 0.5 * (Slope_mean) ** -0.333,
                "NRCS-SCS method": 0.057 * ((1000 / (CN)) - 9) ** 0.7 * L ** 0.8 * (Slope_mean) ** -0.5,
                "Texas Department of transportation’s formula": 0.0369986 * (1.1 - C) * L ** 0.5 * (Slope_mean) ** -0.333,
                "Ventura’s Formula": 0.1272 * A ** 0.5 * (S) ** -0.5,
                "Pasini’s Formula": 0.108 * A ** 0.333 * L ** 0.333 * (Slope_mean) ** -0.5,
                "Tournon’s Formula": 0.396 * L * S ** -0.5 * (A * S ** 0.5 * L ** -2 * Slope_mean ** -0.5) ** 0.72,
                "Puglisi and Zanframundo’s Formula": 6 * L ** 0.666 * DeltaH ** -0.333,
                "Fattorelli and Marchi's Formula": 5.13 * L ** 0.666 * DeltaH ** -0.333,
                "Pezzoli's Formula": 0.055 * L * S ** -0.5,
                "Eaton's Method": 0.796074 * A ** 0.37 * L ** 0.037 * r ** -0.37,
                "Sheridan's Formula": 0.39 * L ** 0.72 * S ** -0.36,
                "Témez's Formula": 0.3 * L ** 0.76 * S ** -0.19,
                "Chow's Model": 0.1602 * L ** 0.64 * S ** -0.32,
                "Dooge's Formula": 0.365 * A ** 0.41 * S ** -0.17,
                "Hatkanir and Sezen": 0.7473 * L ** 0.841,
                "Espey's Method": 0.381088 * L ** 0.36 * S ** -0.18,
                "Pilgrim and McDermott's Formula": 0.76 * A ** 0.38,
                "Ferro's Model": 0.675 * A ** 0.5,
                "Bocchiola's Model": 0.44833 * ((1000 / CN) - 9) ** 0.127 * L ** 0.815 * (Slope_mean) ** -0.216
            }

            return formulas

        except KeyError as e:
            print(f"KeyError: {e}")
            return {}
        except ZeroDivisionError:
            print("Error: Division by zero occurred in formula calculations.")
            return {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}
    
def TC(request):
    """
    View function for the 'Time of Concentration' page.
    """
    view = TimeOfConcentrationView()
    return view.handle_request(request)
