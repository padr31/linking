import pandas as pd
from biopandas.pdb import PandasPdb

# Initialize a new PandasPdb object
ppdb = PandasPdb()
ppdb.read_pdb("./datasets/refined-set/1a1e/1a1e_pocket.pdb")

pd.set_option("display.max_columns", None)
print(ppdb.df["ATOM"].head(3))
