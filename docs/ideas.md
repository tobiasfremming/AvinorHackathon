## Data Columns Available

| Column              | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| `flight_id`         | Flyvningsnummer, første to tegn identifiserer flyselskapet  |
| `dep_airport`       | Avgangsflyplass                                             |
| `dep_airport_group` | Gruppe som avgangsflyplassen tilhører                       |
| `arr_airport`       | Ankomstflyplass                                             |
| `arr_airport_group` | Gruppe som ankomstflyplassen tilhører                       |
| `service_type`      | Flyvingsart, f.eks. rute, charter, frakt, osv               |
| `cancelled`         | 0/1 flagg som indikerer om planlagt flyvning ble kansellert |
| `std`               | Planlagt avgangstid (scheduled time of departure)           |
| `sta`               | Planlagt ankomstid (scheduled time of arrival)              |
| `atd`               | Faktisk avgangstid (actual time of departure)               |
| `ata`               | Faktisk ankomstid (actual time of arrival)                  |

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

Vindreting + vindstyrke sammen med retningen fra flyplass A til flyplass B (medvind eller motvind)

påvirker sykdom concurrency? Det er både mer sykdom om høsten - finne statistisk data om sykdom

streik - finn data om når flyselskap har streik
(admin folk streiker minst, eksterne teknikkere blir ansatt under streik, )

baser blir lagt ned (der flyvertinner og piloter bor. Når det blir nedlagt er det veldig stress for flyvertinner)

Er det noen kommende streik fremover?

på vinter er det mer sbøstorm, det blir glatt på rullebane, kan forsinke eller kansellere fly

Om sommern går siste fly 22 elelrno, om vintern går siste fly klokka 9

latitude og longitude av lufthavn i sin og cos

avstand fra flyplass A til B, lat og long forskjell

data om hvor tett befolket det er i området

Sekvensielle flyvinger

2 forskjellige modeller, en for forsinkelser på bakken, og en for forsinkelser i luften

manglende rullebanebelysning kan gjøre at de stenger ved mørkets frembrudd.

mulig feature: forsinkelsesnivå på hubs

feature: antall fly per dag


Sannsynligvis ikke så kritisk men gøy ting å vurdere: Oil industry helicopter schedules (Stavanger)







Afras Mansoor
  00:52
Også en liten tanke, siden vi har tilgang til alle flyvninger, tror jeg det kan være lurt å se på om samme flyet skal ta flere flyvninger samme dag. Tror om et fly skal fra oslo til bergen til trondheim og viderre igjhen samme dag, øker det sannsynlighet for forsinkelse litt på siste flyplass f.eks
00:52
og bare generelt det at flyvninger er ikke uavhengige
00:54
Og bare ha en feature som er generell trafikk/størrelse eller noe relativt som sier noe om mengden flyvninger per flyplass