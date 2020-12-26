from biopandas.pdb import PandasPdb
import pandas as pd

# Initialize a new PandasPdb object
ppdb = PandasPdb()
ppdb.read_pdb('./datasets/refined-set/1a1e/1a1e_pocket.pdb')

pd.set_option('display.max_columns', None)
print(ppdb.df['ATOM'].head(3))