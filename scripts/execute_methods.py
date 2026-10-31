"""Execute the data-free original-source walkthrough; no model/provider invocation."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'notebooks/METHODS_WALKTHROUGH_EXECUTED.ipynb'
nb=nbformat.read(path,as_version=4)
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
for cell in nb.cells:
    cell.metadata.pop('execution',None)
nbformat.write(nb,path)
print('Executed source inspection only:',path.name)
