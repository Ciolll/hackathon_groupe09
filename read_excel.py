import msoffcrypto
import io
import pandas as pd

def read_excel(file_path, password):    
    with open(file_path, "rb") as f:
        office_file = msoffcrypto.OfficeFile(f)
        office_file.load_key(password=password)
        
        decrypted = io.BytesIO()
        office_file.decrypt(decrypted)
    
    df = pd.read_excel(decrypted, dtype=str)
    return df