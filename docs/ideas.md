
## Data Columns Available
| Column | Description |
|--------|-------------|
| `flight_id` | Flyvningsnummer, første to tegn identifiserer flyselskapet |
| `dep_airport` | Avgangsflyplass |
| `dep_airport_group` | Gruppe som avgangsflyplassen tilhører |
| `arr_airport` | Ankomstflyplass |
| `arr_airport_group` | Gruppe som ankomstflyplassen tilhører |
| `service_type` | Flyvingsart, f.eks. rute, charter, frakt, osv |
| `cancelled` | 0/1 flagg som indikerer om planlagt flyvning ble kansellert |
| `std` | Planlagt avgangstid (scheduled time of departure) |
| `sta` | Planlagt ankomstid (scheduled time of arrival) |
| `atd` | Faktisk avgangstid (actual time of departure) |
| `ata` | Faktisk ankomstid (actual time of arrival) |

## Features to Engineer
- Delay calculations (atd - std, ata - sta)
- Flight duration (actual vs scheduled)
- Seasonal patterns
- Weather impact correlations

## Features to use
- dep_airport
- arr_airpor
- service_type
- cancelled
- std
- sta
- Flight duration (scheduled)
- Season (date as cosine?)
- weather data 
- day of week
- Additional cosine features
- Is holiday

## Target fetures
- atd
- ata
- Delay calculations (atd - std, ata - sta)
- Flight duration (actual)

## Research Tasks
chatgpt deep research er det noen av flyplassende som vil ha nedsatt kapasitet i oktober 2025?
Hente værdata
Airport congestion metrics
Holiday data
Kronekursen, lage et sett på gjennomsnitt av kronekurs 3 måneder i forveien
Corona tiltak


