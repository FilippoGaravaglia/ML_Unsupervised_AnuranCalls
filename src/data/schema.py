MFCC_COLUMNS = [
    f"MFCCs_{index:>2}"
    for index in range(1, 23)
]

LABEL_COLUMNS = [
    "Family",
    "Genus",
    "Species",
]

ID_COLUMN = "RecordID"