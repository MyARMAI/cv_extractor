from glob import glob
import re
import os
import win32com.client as win32
from win32com.client import constants
# convert doc to docx
# Create list of paths to .doc files


def save_as_docx(path):
    # Opening MS Word
    word = win32.gencache.EnsureDispatch('Word.Application')
    doc = word.Documents.Open(path)
    doc.Activate()

    # Rename path with .docx
    new_file_abs = os.path.abspath(path)
    new_file_abs = re.sub(r'\.\w+$', '.docx', new_file_abs)

    # Save and Close
    word.ActiveDocument.SaveAs(
        new_file_abs, FileFormat=constants.wdFormatXMLDocument
    )
    doc.Close(False)


if __name__ == "__main__":
    paths = glob(
        r"C:\Users\Cheikh\Desktop\Projet_memoire\myArmAi\samples\cv_atos\*.doc", recursive=True)

    for path in paths:
        save_as_docx(path)
