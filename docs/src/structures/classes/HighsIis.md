#  [HighsIis](@id highs-iis-class)

Irreducible infeasibility system (IIS) data are communicated via an instance of the `HighsIis` class.

- `valid\_`: The data in the HighsIis instance is valid
- `strategy\_`: The IIS strategy used
- `col\_index\_`: The indices of model columns in the IIS
- `row\_index\_`: The indices of model rows in the IIS
- `col\_bound\_`: The bounds on each column that define the IIS
- `row\_bound\_`: The bounds on each row that define the IIS
- `col\_status\_`: Indicates whether a column in the model is in an IIS, may be in an IIS, or is not in an IIS
- `row\_status\_`: Indicates whether a row in the model is in an IIS, may be in an IIS, or is not in an IIS
- `info\_`: Data on the time and number of simplex iterations required to form the IIS
- `model\_`: A `HighsModel` consisting of the variables, constraints and bounds in the IIS. Currently only its `HighsLp` instance is relevant


