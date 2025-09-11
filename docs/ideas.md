
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

## Features to Engineer/Use
- Delay calculations (atd - std, ata - sta)
- Flight duration (actual vs scheduled)
- Airport congestion metrics
- Seasonal patterns
- Weather impact correlations
- Cancellation prediction features

## Research Tasks
chatgpt deep research er det noen av flyplassende som vil ha nedsatt kapasitet i oktober 2025?

Hente værdata





