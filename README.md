# Avinor Hackathon - Flyforsinkelse Prediksjon

## Oversikt

Dette prosjektet predikerer flyforsinkelser ved norske flyplasser ved hjelp av maskinlæring. Løsningen analyserer historiske flydata, værdata og andre faktorer for å forutsi forsinkelser på ankomst og avgang.

### Kodebase Struktur

```
├── src/                    # Hovedkildekode
│   ├── feature_engineering.py  # Feature engineering pipeline
│   ├── preprocessing.py         # Databehandling
│   ├── holiday_features.py      # Helligdagsfunksjoner
│   └── traffic.py              # Trafikkmønstre
├── feature_engineering/    # Jupyter notebooks for feature utvikling
├── prediction/            # ML modeller og prediksjonsscripts
├── models/               # Lagrede modeller
│   ├── arrival_delay_model/    # Ankomstforsinkelse modell
│   └── departure_delay_model/  # Avgangsforsinkelse modell
├── data/                 # Datasett (gitignored)
├── tests/               # Enhetstester
└── docs/               # Dokumentasjon
```

## Installasjon

1. **Klone repository:**
   ```bash
   git clone <repository-url>
   cd AvinorHackathon
   ```

2. **Sett opp Python miljø:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # eller .venv\Scripts\activate  # Windows
   ```

3. **Installer avhengigheter:**
   ```bash
   pip install -r requirements.txt
   # eller bruk uv:
   uv sync
   ```

