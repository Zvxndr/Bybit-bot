import requests
try:
    response = requests.get('http://localhost:5000/api/historical-data/discover')
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'Success: {data["success"]}')
        print(f'Datasets: {len(data["datasets"])}')
        print(f'Total Records: {data["total_records"]:,}')
        if data["datasets"]:
            for ds in data["datasets"]:
                print(f'  - {ds["symbol"]} {ds["timeframe"]}: {ds["record_count"]:,} records')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Connection error: {e}')