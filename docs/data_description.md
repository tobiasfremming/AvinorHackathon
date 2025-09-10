# Vedlegg 1: Beskrivelse av datasett

I dette vedlegget beskrives datasettene som deles for å ta i bruk i konkurransen.

## Oppgave

Oppgaven går ut på å bygge en maskinlæringsmodell som kan predikere **sannsynlighet for samtidighet** pr flyplassgrupper pr time i perioden **01.10.2025 til 31.10.2025**.

## Datagrunnlag

### Historisk ruteprogram

Datasett med flybevegelser, planlagte og faktiske avgangs- og landingstider, flyplasser, flyselskap, flyvningstype, mm. Datasettet utleveres som CSV.

**Fil**: `historical_flights.csv`

| Kolonne | Beskrivelse |
|---------|-------------|
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
### Treningsdata

Vi har brukt datasettet over (`historical_flights.csv`) til å lage et enkelt datasett for trening. I vårt treningssett har vi laget et par enkle features til inspirasjon. Vi oppfordrer deltakere til å lage egne features, både basert på ruteprogrammet og på andre åpne datakilder, for eksempel værdata, helligdager, og annet. Vi tror at det er mye å hente i å være kreativ her.

**Fil**: `training_data.csv`

| Kolonne | Beskrivelse |
|---------|-------------|
| `airport_group` | Se tabellen `airportgroups.csv` for å se hvilke flyplasser hører til hvilken gruppe |
| `date` | Dato |
| `hour` | Time, f.eks. gjelder verdien 9 for tidsrommet 09:00-09:59 |
| `target` | 0/1-flagg som indikerer om det har skjedd samtidighet basert på faktiske flytider |
| `feat_season` | Årstid |
| `feat_sched_concurrence` | 0/1-flagg for antatt samtidighet basert på planlagte flytider |
### Prediksjonsdata

Foreløpig ruteprogram for oktober 2025 legges ut sammen med oppgaven.

> **OBS!**: Oppdatert ruteprogram for oktober vil legges ut **26. september, kl. 09.00**.

**Fil**: `schedule_oct2025.csv`

| Kolonne | Beskrivelse |
|---------|-------------|
| `flight_id` | Flyvningsnummer, første to tegn identifiserer flyselskapet |
| `dep_airport` | Avgangsflyplass |
| `dep_airport_group` | Gruppe som avgangsflyplassen tilhører |
| `arr_airport` | Ankomstflyplass |
| `arr_airport_group` | Gruppe som ankomstflyplassen tilhører |
| `service_type` | Flyvingsart, f.eks. rute, charter, frakt, osv |
| `std` | Planlagt avgangstid (scheduled time of departure) |
| `sta` | Planlagt ankomstid (scheduled time of arrival) |

Dette datasettet kan brukes til å lage data som dere skal gjøre prediksjon på. Vi har laget et eksempel på dette her:

**Fil**: `inference_data_oct2025.csv`

| Kolonne | Beskrivelse |
|---------|-------------|
| `airport_group` | Se tabellen `airportgroups.csv` for å se hvilke flyplasser hører til hvilken gruppe |
| `date` | Dato |
| `hour` | Time, f.eks. gjelder verdien 9 for tidsrommet 09:00-09:59 |
| `target` | 0/1 flagg som indikerer om det har vært samtidighet i det aktuelle tidsrommet |
| `feat_season` | Årstid |
| `feat_sched_concurrence` | 0/1-flagg for antatt samtidighet basert på planlagte flytider |