import hashlib

monasterium_url_root = "https://www.monasterium.net/mom/"

def chatomid_to_url(atomid, root=monasterium_url_root):
    parts = atomid.split("/")                                                                                                                                  
    if len(parts) == 5:
        return f"{root}{parts[2]}/{parts[3]}/{parts[4]}/charter"
    elif len(parts) == 4:
        return f"{root}{parts[2]}/{parts[3]}/charter"
    else:
        raise ValueError("Invalid atom_id length.")
    
def to_md5(string, trunc_threshold=16): 
    md5sum = hashlib.md5(string.encode("utf-8")).hexdigest()[trunc_threshold:]
    return md5sum