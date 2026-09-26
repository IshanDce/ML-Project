#!/usr/bin/env python3
"""
src/cleaner.py
High-Performance Text Normalization & Transliteration Module for Business Entity Resolution.
Supports: English, Hindi (Devanagari phonetic & dictionary transliteration), French (accents & forms).
"""
import re
import unicodedata

# 1. Hindi Business Keyword Replacements (Dictionary mapping)
HINDI_BUSINESS_DICT = {
    "प्राइवेट": "private",
    "प्रा.": "pvt",
    "प्रा": "pvt",
    "लिमिटेड": "limited",
    "लि.": "ltd",
    "लि": "ltd",
    "इंटरप्राइजेज": "enterprises",
    "एंटरप्राइजेज": "enterprises",
    "एलएलपी": "llp",
    "टेक्नोलॉजीज": "technologies",
    "सॉल्यूशंस": "solutions",
    "सॉल्यूशन": "solutions",
    "सर्विसेज": "services",
    "सर्विस": "services",
    "सिस्टम्स": "systems",
    "सिस्टम": "systems",
    "इम्पेक्स": "impex",
    "इंडस्ट्रीज": "industries",
    "इंडस्ट्री": "industry",
    "मार्केटिंग": "marketing",
    "फाइनेंस": "finance",
    "प्रॉपर्टीज": "properties",
    "कॉर्पोरेशन": "corporation",
    "एसोसिएट्स": "associates",
    "ट्रेडर्स": "traders",
    "ट्रेडिंग": "trading",
    "कंसल्टेंट्स": "consultants",
    "कंसल्टेंसी": "consultancy",
    "कंपनी": "company",
    "ग्रुप": "group",
    "हॉस्पिटल्स": "hospitals",
    "हॉस्पिटल": "hospital",
    "फार्मा": "pharma",
    "फूड": "food",
    "फूड्स": "foods",
    "ऑटो": "auto",
    "मोटर्स": "motors",
    "लॉजिस्टिक्स": "logistics",
    "इंफ्रास्ट्रक्चर": "infrastructure",
    "इंफ्राटेक": "infratech",
    "सिक्योरिटीज": "securities",
    "सिक्योरिटी": "security",
    "डेवलपर्स": "developers",
    "इंजीनियरिंग": "engineering",
    "इंडिया": "india",
    "भारत": "bharat",
    "श्री": "shree",
    "श्रीराम": "shriram",
    "साईं": "sai",
    "साई": "sai",
    "ओम": "om",
    "जय": "jai",
    "न्यू": "new",
    "सुपर": "super",
    "रॉयल": "royal",
    "ग्लोबल": "global",
    "प्राइम": "prime",
    "सेवन": "seven",
    "वन": "one",
    "विजन": "vision",
    "आदित्य": "aditya",
    "राम": "ram",
    "मॉडर्न": "modern",
}

# 2. Hindi Devanagari Character-to-Latin Phonetic Map
DEVANAGARI_CHARS = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    # Consonants
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh', 'फ़': 'f',
    # Matras
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'ं': 'n', 'ँ': 'n', 'ः': 'h', '्': '',
    # Digits
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}

def transliterate_devanagari(text):
    """Transliterate Devanagari Hindi text into Latin/English representation."""
    if not any(0x0900 <= ord(c) <= 0x097F for c in text):
        return text
    
    # First replace known business words
    words = text.split()
    translated_words = []
    for w in words:
        cleaned_w = w.strip(".,/-()[]")
        if cleaned_w in HINDI_BUSINESS_DICT:
            translated_words.append(HINDI_BUSINESS_DICT[cleaned_w])
        else:
            # Phonetic character replacement
            res = []
            for ch in w:
                res.append(DEVANAGARI_CHARS.get(ch, ch))
            translated_words.append("".join(res))
    return " ".join(translated_words)

# 3. URL and Web Suffix Patterns
URL_PATTERN = re.compile(
    r'\b(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+)\.(?:com|org|net|in|co\.in|fr|io|biz|info|co|edu|gov)\b',
    re.IGNORECASE
)

# 4. Standard Legal Suffix Canonicalization
LEGAL_EXPANSIONS = [
    (re.compile(r'\b(private\s+limited|pvt\.?\s*ltd\.?|p\.?\s*ltd\.?|pvt\s+limited)\b', re.I), ' pvt ltd '),
    (re.compile(r'\b(limited\s+liability\s+company|l\.?l\.?c\.?)\b', re.I), ' llc '),
    (re.compile(r'\b(limited\s+liability\s+partnership|l\.?l\.?p\.?)\b', re.I), ' llp '),
    (re.compile(r'\b(corporation|corp\.?)\b', re.I), ' corp '),
    (re.compile(r'\b(incorporated|inc\.?)\b', re.I), ' inc '),
    (re.compile(r'\b(limited|ltd\.?)\b', re.I), ' ltd '),
    (re.compile(r'\b(company|co\.?)\b', re.I), ' co '),
    (re.compile(r'\b(societe\s+a\s+responsabilite\s+limitee|s\.?a\.?r\.?l\.?)\b', re.I), ' sarl '),
    (re.compile(r'\b(societe\s+par\s+actions\s+simplifiee|s\.?a\.?s\.?)\b', re.I), ' sas '),
    (re.compile(r'\b(societe\s+civile\s+immobiliere|s\.?c\.?i\.?)\b', re.I), ' sci '),
    (re.compile(r'\b(societe\s+anonyme|s\.?a\.?)\b', re.I), ' sa '),
]

LEGAL_SUFFIX_TOKENS = {
    'pvt', 'ltd', 'llc', 'llp', 'inc', 'corp', 'co', 'sarl', 'sas', 'sci', 'sa',
    'enterprises', 'enterprise', 'services', 'service', 'solutions', 'solution',
    'technologies', 'technology', 'tech', 'group', 'holdings', 'holding', 'international',
    'consultants', 'consulting', 'industries', 'industry', 'associates'
}

# 5. Address Standardization
ADDR_EXPANSIONS = [
    (re.compile(r'\b(st\.?|street)\b', re.I), ' st '),
    (re.compile(r'\b(rd\.?|road)\b', re.I), ' rd '),
    (re.compile(r'\b(ave\.?|av\.?|avenue)\b', re.I), ' ave '),
    (re.compile(r'\b(blvd\.?|boulevard)\b', re.I), ' blvd '),
    (re.compile(r'\b(dr\.?|drive)\b', re.I), ' dr '),
    (re.compile(r'\b(hwy\.?|highway)\b', re.I), ' hwy '),
    (re.compile(r'\b(pl\.?|place)\b', re.I), ' pl '),
    (re.compile(r'\b(ct\.?|court)\b', re.I), ' ct '),
    (re.compile(r'\b(sq\.?|square)\b', re.I), ' sq '),
    (re.compile(r'\b(ln\.?|lane)\b', re.I), ' ln '),
    (re.compile(r'\b(ste\.?|suite|unit|apt\.?|apartment)\b', re.I), ' ste '),
    (re.compile(r'\b(sec\.?|sector)\b', re.I), ' sector '),
    (re.compile(r'\b(bldg\.?|building)\b', re.I), ' bldg '),
    (re.compile(r'\b(fl\.?|floor)\b', re.I), ' fl '),
    (re.compile(r'\b(h\.?no\.?|house\s+no\.?)\b', re.I), ' hno '),
    (re.compile(r'\b(r\.?|rue)\b', re.I), ' rue '),
    (re.compile(r'\b(bd\.?|boulevard)\b', re.I), ' blvd '),
]

STATE_MAP = {
    'ohio': 'oh', 'california': 'ca', 'texas': 'tx', 'florida': 'fl', 'new york': 'ny',
    'pennsylvania': 'pa', 'illinois': 'il', 'north carolina': 'nc', 'oklahoma': 'ok',
    'arizona': 'az', 'washington': 'wa', 'georgia': 'ga', 'michigan': 'mi', 'virginia': 'va',
    'district of columbia': 'dc', 'west virginia': 'wv',
    'west bengal': 'wb', 'uttar pradesh': 'up', 'madhya pradesh': 'mp', 'maharashtra': 'mh',
    'tamil nadu': 'tn', 'karnataka': 'ka', 'gujarat': 'gj', 'rajasthan': 'rj', 'haryana': 'hr',
}

def strip_accents(text):
    """Normalize unicode and strip diacritical marks (e.g., French accents)."""
    return "".join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

def clean_business_name(name):
    """
    Cleans and normalizes a business name.
    Returns:
        clean_name: full normalized name
        core_name: name with legal entity suffixes and generic stopwords removed
        first_token: the primary identifying token (or first 2 tokens)
    """
    if not name:
        return "", "", ""
    
    # 1. Transliterate Devanagari if present
    text = transliterate_devanagari(name)
    
    # 2. Normalize accents
    text = strip_accents(text)
    
    # 3. Handle URLs (e.g. strategicpraetorian.com -> strategic praetorian)
    text = URL_PATTERN.sub(r' \1 ', text)
    
    # 4. Symbol replacements
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    text = text.replace('+', ' plus ')
    
    # 5. Remove leading/trailing non-alphanumeric noise (-- Holloway -> Holloway, << Team -> Team)
    text = re.sub(r'^[\W_]+|[\W_]+$', '', text)
    
    # 6. Lowercase & replace punctuation with space
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 7. Standardize legal suffixes
    for pattern, repl in LEGAL_EXPANSIONS:
        text = pattern.sub(repl, text)
        
    tokens = [t for t in text.split() if t]
    clean_name = " ".join(tokens)
    
    # 8. Core name (strip common legal suffixes and stopwords)
    core_tokens = [t for t in tokens if t not in LEGAL_SUFFIX_TOKENS]
    if not core_tokens:
        core_tokens = tokens  # fallback if all were legal tokens
    core_name = " ".join(core_tokens)
    
    # 9. First significant token
    first_token = core_tokens[0] if core_tokens else (tokens[0] if tokens else "")
    
    return clean_name, core_name, first_token

def clean_address(addr):
    """
    Cleans and normalizes business address.
    Extracts numbers (zipcodes, street numbers) and standardized tokens.
    """
    if not addr:
        return "", set()
    
    # 1. Devanagari transliterate & strip accents
    text = transliterate_devanagari(addr)
    text = strip_accents(text)
    
    # 2. Lowercase and basic punctuation
    text = text.lower()
    text = text.replace('&', ' and ')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # 3. Address abbreviations
    for pattern, repl in ADDR_EXPANSIONS:
        text = pattern.sub(repl, text)
        
    # 4. State normalization
    for full_state, abbrev in STATE_MAP.items():
        text = re.sub(rf'\b{full_state}\b', f' {abbrev} ', text)
        
    tokens = [t for t in text.split() if t]
    clean_addr = " ".join(tokens)
    
    # 5. Extract numeric tokens (street numbers, zipcodes)
    numbers = set(re.findall(r'\b\d+\b', clean_addr))
    
    return clean_addr, numbers

if __name__ == "__main__":
    # Test suite on actual examples seen in inspection
    test_cases = [
        ("Strategic Praetorian", "702 N Street, Washington, DC"),
        ("Strategic Praetorian Inc", "N ST, WASHINGTON, DC"),
        ("strategicpraetorian.com", "702 1/2 N St, Washington, District of Columbia"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"),
        ("-- Holloway Peak Inc Seafood", "105 ELM ST, MORGANTON, NC"),
        ("आदित्य प्रॉपर्टीज एलएलपी", "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh"),
        ("Prime Seven Impex Private Limited", "49-A Dipak Nagar Patel Nagar, Indra Nagar, Lucknow, Uttar Pradesh"),
        ("प्राइम सेवन इम्पेक्स प्राइवेट लिमिटेड", "INDRA NAGAR, उत्तर प्रदेश, NO 49-A DIPAK NAGAR PATEL NAGAR, LUCKNOW"),
        ("SCI Ptit Àmicale", "18 RUE JEN ZAY, Dunkerque, Nord"),
        ("Fractales Amis Groupe S.A.S", "23 Rue Icmre, La Teste-de-buch, Gironde"),
        ("Chi1ders Ampercap Ampercap Inc", "32C Anna Street, Dayton CITY, Ohio"),
    ]
    
    print("=== TESTING CLEANER ON REAL EXAMPLES ===")
    for name, addr in test_cases:
        c_name, core_name, f_tok = clean_business_name(name)
        c_addr, nums = clean_address(addr)
        print(f"Orig Name:   {name}")
        print(f"  Clean:     {c_name}")
        print(f"  Core:      {core_name} | FirstTok: {f_tok}")
        print(f"Orig Addr:   {addr}")
        print(f"  CleanAddr: {c_addr}")
        print(f"  Numbers:   {sorted(nums)}")
        print("-" * 60)
