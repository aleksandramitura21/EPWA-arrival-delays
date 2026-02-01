# EPWA ATFM Arrival Delay Prediction (AWS + Spark)

Projekt analityczno-predykcyjny dotyczący opóźnień przylotowych ATFM
dla lotniska Warszawa–Chopin (EPWA), zrealizowany w środowisku chmurowym AWS
z wykorzystaniem Apache Spark.
Celem projektu jest analiza historycznych opóźnień ATFM, budowa modelu
predykcyjnego oraz wykonanie scenariuszowej prognozy opóźnień.

---

## Cel projektu

- analiza danych ATFM Arrival Delay dla lotniska EPWA
- eksploracyjna analiza danych (EDA)
- identyfikacja anomalii operacyjnych
- budowa modelu regresyjnego (Random Forest)
- walidacja i interpretacja wyników
- prognoza opóźnień na rok 2026

---

## Architektura rozwiązania

Projekt wykorzystuje klasyczną architekturę warstwową:
Bronze → Silver → Gold

- Bronze – surowe pliki CSV (S3)
- Silver – dane dzienne po ETL (Parquet)
- Gold – EDA, model, walidacja, prognozy, wykresy

Technologie:
- AWS S3
- AWS EC2
- Apache Spark
- Spark MLlib
- Athena (analiza wyników)
- Python

---

## Struktura projektu

scripts/
├── etl.py                  # ETL: CSV → Silver (daily)
├── eda_basic.py            # Podstawowa EDA
├── eda_anomalies.py        # Analiza anomalii
├── model_v1.py             # Model Random Forest
├── model_validation.py     # Walidacja modelu
├── model_interpretation.py # Interpretacja modelu
├── forecast_2026.py        # Prognoza opóźnień 2026
└── report_export.py       # Wykresy do raportu
raport/
└──  Raport.pdf             # Raport z projektu
---

## Dane

Dane pochodzą z systemu *EUROCONTROL – Airport Arrival ATFM Delay*.

Zakres danych:
- opóźnienia ATFM
- dane dzienne
- podział na kategorie przyczyn (causes)
- lata historyczne: 2014–2024

---

## Model

- typ: regresja
- algorytm: Random Forest Regressor
- zmienna objaśniana: AVG_ARR_DELAY_MIN
- cechy:
  - natężenie ruchu
  - struktura przyczyn opóźnień
  - cechy kalendarzowe

Metryki:
- RMSE
- R²
- walidacja miesięczna

---

## Prognoza

Wykonano scenariuszową prognozę opóźnień na rok 2026
na podstawie baseline z lat 2022–2024.

Prognoza ma charakter:
- długookresowy
- trendowy
- nieoperacyjny (bez zdarzeń losowych)

---

## Ograniczenia

- dane obejmują wyłącznie opóźnienia ATFM
- brak rzeczywistych danych meteorologicznych
- brak informacji o przyszłych zmianach infrastrukturalnych

---

## Licencja

Projekt udostępniony na licencji MIT.
