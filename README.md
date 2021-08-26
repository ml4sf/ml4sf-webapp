# ml4sf-webapp

- `molecular_properties_calculation.py` depends on `numpy, pandas and pytorch`
- There are several TODOS tthat I will address later

The code is supposed to work as follows:

- The constructor to `PropertiesCalculation` takes the contents of a `*.mdl` file as a string
- The `run_calculations` method runs all calcuations and returns the properties 
  as a Python dictionary (hashmap)
