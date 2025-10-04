# Feature Catalog

This document summarises the predictors used by the analgesia models. All features are derived from prehospital data unless noted otherwise.

## Numeric Features

- AF
- Alter 
- GCS
- HF
- HR
- IBD Diastolisch
- IBD Systolisch
- Körperfläche
- NACA
- NACA (nummerisch)
- NIBD Diastolisch
- NIBD Systolisch
- PLZ
- SPO2
- Temperatur
- VAS_on_scene
- etCO2

## Categorical Features

- (Be)-Atmung
- Aktuelles Ereignis
- Alle ICD-Codes
- Arm (Bewegung li)
- Arm (Bewegung re)
- Arm (Empfindlichkeit li)
- Arm (Empfindlichkeit re)
- Arme
- Atemgeräusche
- Atemwegbefund
- Auffälligkeiten
- Beatmung
- Befund
- Befund Atmung
- Bein (Bewegung li)
- Bein (Bewegung re)
- Bein (Empfindlichkeit li)
- Bein (Empfindlichkeit re)
- Bergungen
- Bewusstseinlage
- Cincinnati FAST
- Detail
- EKG 12-Kanal
- EKG 3-Kanal
- EKG Telemetrie
- Einteilung
- Ereignisort2
- Geschlecht
- Gesicht
- Hautbefund
- Herzrhytmusstörungen
- ICD-Code der Hauptdiagnose
- Institution
- Ist ossär bestehend
- Ist venös bestehend
- Kanton
- Kategorie
- Kategorie (reduziert)
- Körperregion
- Lagerungen
- Lebensbedrohliche Blutung
- Lichtreaktion (re)
- Monat
- Ort
- Ort3
- Pupillenweite (li)
- Pupillenweite (re)
- Sauerstoffabgaben
- Sprache
- Strasse
- Tag oder Nacht
- Weitere Diagnosen
- Weitere Massnahmen
- Wochentag
- Zeitpunkt, Messart, Wert
- psychischer Erstbefund

## Multi-label Text Features

- Blutstillung
- Ereignisort
- Puls tastbar
- Thoraxdrainage
- Zugänge

## Engineered Features

- PLZ (combined from PLZ/PLZ4)
- doctor_age (reference year 2025)
- doctor_sex (mapped from roster)
- doctor_specialist_qualifications (from metadata)
