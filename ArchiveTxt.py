import requests
ROWS = 2
PAGES = 200
PAGE = 201

def fetch_archive_texts_by_subject(subject, collections=None, rows=ROWS, pages=(PAGES+PAGE), page=PAGE):
    """
    Fetch all available .txt texts from archive.org
    matching a subject and optional list of collections,
    and combine them into one long string separated by spaces.

    Parameters:
        subject (str): Subject term to search.
        collections (list[str] or None): List of collection names.
        rows (int): Number of results per query batch.
        delay (float): Delay between requests.

    Returns:
        str: One long string containing all text separated by spaces.
    """

    base_search = "https://archive.org/advancedsearch.php"
    long_text = []

    # Build query
    query = f'subject:"{subject}" AND language:eng'
    if collections:
        collection_query = " OR ".join([f'collection:{c}' for c in collections])
        query += f" AND ({collection_query})"
    params = {
        "q": query,
        "fl[]": ["identifier"],
        "rows": rows,
        "page": PAGE,
        "output": "json"
    }
    print(f"{subject} page: ",end="", flush=True)
    while params["page"] <= pages:
        print(f"{params['page']} ",end="", flush=True)
        response = requests.get(base_search, params=params)
        data = response.json()

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            break

        for doc in docs:
            identifier = doc["identifier"]

            # Fetch metadata to find .txt files
            meta_url = f"https://archive.org/metadata/{identifier}"
            meta_resp = requests.get(meta_url)
            if meta_resp.status_code != 200:
                continue

            meta_json = meta_resp.json()
            files = meta_json.get("files", [])

            # Look for .txt files
            txt_files = [f["name"] for f in files if f["name"].lower().endswith(".txt")]

            for txt_file in txt_files:
                txt_url = f"https://archive.org/download/{identifier}/{txt_file}"
                txt_response = requests.get(txt_url)
                if txt_response.status_code == 200:
                    text_content = txt_response.text
                    # replace newlines with spaces
                    text_content = " ".join(text_content.split())
                    long_text.append(text_content)
        params["page"] += 1
    print()

    return " ".join(long_text)
    
    
s = "Short stories"#["Philosophy", "Novels", "Short stories"]
s = "Philosophy"#["Philosophy", "Novels", "Short stories"]
#s = "Poetry"
#s = "Logic"
#s = "Reasoning"
text = fetch_archive_texts_by_subject(
    subject=s,
    collections=["gutenberg", "americana"]
)

file_name = f"{s}_ArchiveTxt_ROWS_{ROWS}_PAGES_{PAGES}_START_PAGE_{PAGE}.txt"
with open(file_name, "w", encoding="utf-8") as f:
    f.write(f"{text}")
    print(f"done, check: {file_name}")

subject1 = "Philosophy"#["Philosophy", "Novels", "Short stories"]
subject2 = "Fiction"#["Fiction", "Ethics", "Metaphys*"]

