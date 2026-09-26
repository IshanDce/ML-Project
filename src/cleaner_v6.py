#!/usr/bin/env python3
"""
src/cleaner_v6.py
High-Performance Multiprocess Cleaner & Normalizer for Entity Resolution.
Extracts normalized names, sorted tokens, first content words, address tokens, numbers, and FKAs.
"""
import time
import re
from unidecode import unidecode
from multiprocessing import Pool
import duckdb

INDIC_EXTRA = {
    'bonmbe': 'bombay', 'bombai': 'bombay', 'bambai': 'bombay',
    'nonrth': 'north', 'nort': 'north',
    'impeks': 'impex',
    'eksportts': 'exports', 'ekspoortts': 'exports', 'eksport': 'exports',
    'ttredding': 'trading', 'treding': 'trading', 'treddar': 'traders',
    'bestt': 'best',
    'grett': 'great', 'gret': 'great',
    'sisttms': 'systems', 'sistams': 'systems', 'sistam': 'system',
    'injiniiyring': 'engineering', 'injiiniyring': 'engineering', 'injiniyaring': 'engineering',
    'praaivett': 'pvt', 'praaibhett': 'pvt', 'piraiveett': 'pvt', 'praivet': 'pvt',
    'limittedd': 'ltd', 'limittett': 'ltd', 'limitet': 'ltd',
    'bluu': 'blue', 'aannd': 'anand', 'arihnt': 'arihant',
    'inphraa': 'infra', 'globl': 'global', 'globaal': 'global',
    'tiruppti': 'tirupati', 'tirupathi': 'tirupati',
    'tekno': 'techno', 'teknoloji': 'technology', 'teknolojiij': 'technologies',
    'inddsttriij': 'industries', 'inddsttri': 'industries',
    'kemikl': 'chemical', 'kemikls': 'chemicals',
    'phaarmaa': 'pharma', 'faarmaa': 'pharma',
    'enterpraaizez': 'enterprises', 'entarpraaiz': 'enterprises',
    'soolhyuusNns': 'solutions', 'solyushans': 'solutions',
    'korhporeSHn': 'corporation', 'kaaraporeSHan': 'corporation',
    'prphektt': 'perfect', 'shrii': 'shree', 'srii': 'shree',
    'ddriim': 'dream', 'om': 'om', 'sai': 'sai', 'saaii': 'sai',
    'phlstiks': 'plastics', 'faainenss': 'finance', 'faainens': 'finance',
    'sekyhurittii': 'security', 'sekyhuriti': 'security',
}

TITLE_PREFIX_RE = re.compile(r'^\s*(the|dr|m/s|mr|ms|shri|smt|sri|messrs)\b[\s\.\-\/]*', re.IGNORECASE)

LEGAL_SUFFIXES = {
    'the', 'and', 'inc', 'incorporated', 'corp', 'corporation', 'llc', 'llp', 'ltd', 'limited',
    'co', 'company', 'enterprises', 'services', 'solutions', 'group', 'technologies',
    'pvt', 'private', 'sarl', 'sas', 'sci', 'sa', 'eurl', 'plc', 'pllc', 'pc', 'lp',
    'associates', 'consultants', 'consulting', 'industries', 'international', 'intl', 'global',
    'traders', 'trading', 'marketing', 'agencies', 'agency', 'holdings', 'ventures', 'investments',
    'foundation', 'trust', 'constructions', 'construction', 'developers', 'development',
    'manufacturers', 'manufacturing', 'properties', 'property', 'logistics', 'transport',
    'exports', 'export', 'imports', 'import', 'exim', 'distributors', 'retail', 'wholesale',
    'pharma', 'foods', 'infra', 'infotech', 'mr', 'ms', 'm/s', 'shri', 'smt', 'sri',
    'center', 'centre', 'care', 'service', 'national',
    'com', 'org', 'net', 'in', 'fr',
}

LEGAL_RE = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)

ADDR_STOPWORDS = {
    'street','st','road','rd','avenue','ave','drive','dr','lane','ln','boulevard','blvd',
    'way','circle','cir','court','ct','floor','fl','suite','ste','unit','apt','apartment',
    'block','plot','house','building','bldg','near','opp','behind','beside','hno','no',
    'sec','sector','phase','village','dist','district','post','po','state','null','none','na',
    'india','delhi','mumbai','bangalore','pune','chennai','hyderabad','kolkata','us','usa',
    'maharashtra','rajasthan','tamil','nadu','karnataka','telangana','uttar','pradesh',
    'west','bengal','gujarat','madhya','haryana','punjab','bihar','kerala','odisha',
}

OCR_TRANS = str.maketrans('0135', 'olse')

def clean_single_record(row):
    eid, bname, baddr, country = row
    bname = bname or ''
    baddr = baddr or ''
    country = country or ''
    
    # Strip URL
    bname_sub = re.sub(r'https?://\S+|www\.\S+|\|\s*www\.[^\s|]+|\.(com|org|net|in|co)\b', ' ', bname, flags=re.IGNORECASE)
    
    # Check fka / aka / dba
    fka_split = re.split(r'\b(?:f/?k/?a|a/?k/?a|d/?b/?a|formerly known as|doing business as)\b', bname_sub, flags=re.IGNORECASE)
    primary_name = fka_split[0]
    fka_name = fka_split[1] if len(fka_split) > 1 else ''
    
    def process_name(s):
        if not s: return '', '', '', '', ''
        if not s.isascii(): s = unidecode(s)
        s = s.lower().replace('&', ' and ')
        s = re.sub(r'[\[\]\(\)\-\+\*\/\\\|\@\#\$\%\^\&]', ' ', s)
        s = TITLE_PREFIX_RE.sub('', s)
        s = s.translate(OCR_TRANS)
        raw_toks = re.findall(r'[a-z0-9]+', s)
        toks = []
        for t in raw_toks:
            t = INDIC_EXTRA.get(t, t)
            t = t.replace('ks', 'x').replace('tt', 't').replace('dd', 'd').replace('pp', 'p')
            t = t.replace('aa', 'a').replace('ee', 'e').replace('ii', 'i').replace('oo', 'o').replace('uu', 'u')
            toks.append(t)
        cname = ' '.join(toks)
        
        c_no_legal = LEGAL_RE.sub('', cname)
        nname = re.sub(r'[^a-z0-9]', '', c_no_legal)
        if len(nname) < 2:
            nname = re.sub(r'[^a-z0-9]', '', cname)
            
        stoks_list = sorted(set(t for t in re.findall(r'[a-z0-9]+', c_no_legal) if len(t) >= 2))
        if not stoks_list:
            stoks_list = sorted(set(t for t in re.findall(r'[a-z0-9]+', cname) if len(t) >= 2))
        stoks = ' '.join(stoks_list)
        
        fcw = ''
        for t in toks:
            if t not in LEGAL_SUFFIXES and len(t) >= 3:
                fcw = t
                break
        if not fcw and toks:
            fcw = toks[0]
            
        pfx4 = nname[:4] if len(nname) >= 4 else nname
        return cname, nname, stoks, fcw, pfx4
    
    cname, nname, stoks, fcw, pfx4 = process_name(primary_name)
    fka_cname, fka_nname, fka_stoks, fka_fcw, fka_pfx4 = process_name(fka_name) if fka_name else ('', '', '', '', '')
    
    addr_str = baddr.lower()
    if not addr_str.isascii(): addr_str = unidecode(addr_str)
    
    nums = re.findall(r'\b\d+\b', addr_str)
    nums = [n.lstrip('0') or '0' for n in nums if len(n.lstrip('0')) >= 1]
    num1 = nums[0] if len(nums) > 0 else ''
    num2 = nums[1] if len(nums) > 1 else ''
    
    awords = [w for w in re.findall(r'[a-z]{4,}', addr_str) if w not in ADDR_STOPWORDS and not w.isdigit()]
    w1 = awords[0] if len(awords) > 0 else ''
    w2 = awords[1] if len(awords) > 1 else ''
    w3 = awords[2] if len(awords) > 2 else ''
    
    return (eid, cname, addr_str, country, nname, stoks, fcw, pfx4, num1, num2, w1, w2, w3,
            fka_nname, fka_stoks, fka_fcw)
